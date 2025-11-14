from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security, HTTPException
import os
import torch
import uuid
import gc
import tempfile
import logging
import sys
from aquilesimage.configs import load_config_app
from typing import Optional
import requests
from PIL import Image as PILImage
import io
import base64
import time

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    
        'INFO': '\033[32m',     
        'WARNING': '\033[33m',  
        'ERROR': '\033[31m',    
        'CRITICAL': '\033[35m', 
        'RESET': '\033[0m',     
        'BOLD': '\033[1m',      
    }
    
    LOGGER_COLOR = '\033[94m'  
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        
        original_format = super().format(record)
        
        colored_message = (
            f"{self.COLORS['BOLD']}{self.LOGGER_COLOR}[{record.name}]{self.COLORS['RESET']} "
            f"{log_color}{record.levelname}{self.COLORS['RESET']}: "
            f"{original_format.split(': ', 1)[1] if ': ' in original_format else record.getMessage()}"
        )
        
        return colored_message

def setup_colored_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(colored_formatter)
    
    logger.addHandler(console_handler)
    
    return logger

logger_utils = setup_colored_logger("Aquiles-Image-Utils", logging.WARNING)

security = HTTPBearer()

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    configs = await load_config_app()

    valid_keys = [k for k in configs["allows_api_keys"] if k and k.strip()]
    
    if not valid_keys:
        return None

    if not credentials:
        raise HTTPException(
            status_code=403,
            detail="API key missing",
        )
    
    api_key = credentials.credentials
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key

class Utils:
    def __init__(self, host: str = '0.0.0.0', port: int = 8500):
        self.service_url = f"http://{host}:{port}"
        self.image_dir = os.path.join(tempfile.gettempdir(), "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.video_dir = os.path.join(tempfile.gettempdir(), "videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def save_image(self, image):
        if hasattr(image, "to"):
            try:
                image = image.to("cpu")
            except Exception:
                pass

        if isinstance(image, torch.Tensor):
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            image = to_pil(image.squeeze(0).clamp(0, 1))

        filename = "img" + str(uuid.uuid4()).split("-")[0] + ".png"
        image_path = os.path.join(self.image_dir, filename)
        logger_utils.warning(f"Saving image to {image_path}")

        image.save(image_path, format="PNG", optimize=True)

        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return os.path.join(self.service_url, "images", filename)

def load_dev_mode_image(DEV_MODE_IMAGE_PATH, DEV_MODE_IMAGE_URL) -> Optional[PILImage.Image]:
    logger = setup_colored_logger("Utils-Dev", logging.INFO)
    try:
        if DEV_MODE_IMAGE_PATH and os.path.exists(DEV_MODE_IMAGE_PATH):
            logger.info(f"Loading dev mode image from local path: {DEV_MODE_IMAGE_PATH}")
            img = PILImage.open(DEV_MODE_IMAGE_PATH)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        
        logger.info(f"Downloading dev mode image from URL: {DEV_MODE_IMAGE_URL}")
        response = requests.get(DEV_MODE_IMAGE_URL, timeout=10)
        response.raise_for_status()
        
        img = PILImage.open(io.BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    except Exception as e:
        logger.warning(f"Failed to load dev mode image: {e}")
        logger.info("Creating placeholder image for dev mode")
        img = PILImage.new('RGB', (1024, 1024), color=(73, 109, 137))
        return img

def create_dev_mode_response(
    DEV_MODE_IMAGE_PATH, 
    DEV_MODE_IMAGE_URL,
    n: int,
    response_format: str,
    output_format: str,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    background: Optional[str] = None,
    utils_app = None,
) -> dict:
    dev_image = load_dev_mode_image(DEV_MODE_IMAGE_PATH, DEV_MODE_IMAGE_URL)
    
    if size:
        if size == "1024x1024":
            target_size = (1024, 1024)
        elif size == "1536x1024":
            target_size = (1536, 1024)
        elif size == "1024x1536":
            target_size = (1024, 1536)
        elif size == "256x256":
            target_size = (256, 256)
        elif size == "512x512":
            target_size = (512, 512)
        elif size == "1792x1024":
            target_size = (1792, 1024)
        elif size == "1024x1792":
            target_size = (1024, 1792)
        else:
            target_size = (1024, 1024)
        
        if dev_image.size != target_size:
            dev_image = dev_image.resize(target_size, PILImage.Resampling.LANCZOS)
    
    images_data = []
    
    for _ in range(n):
        image_obj = {}
        
        if response_format == "b64_json":
            buffer = io.BytesIO()
            dev_image.save(buffer, format=output_format.upper())
            img_str = base64.b64encode(buffer.getvalue()).decode()
            image_obj["b64_json"] = img_str
        else:
            url = utils_app.save_image(dev_image)
            image_obj["url"] = url
        
        images_data.append(image_obj)
    
    response_data = {
        "created": int(time.time()),
        "data": images_data,
    }
    
    if size:
        response_data["size"] = size
    if quality:
        response_data["quality"] = quality
    if background:
        response_data["background"] = background
    if output_format:
        response_data["output_format"] = output_format
    
    return response_data


def create_dev_mode_video_response(
    model: str = "sora-2",
    prompt: Optional[str] = None,
    size: Optional[str] = None,
    seconds: Optional[str] = None,
    quality: Optional[str] = None,
    status: str = "completed",
    progress: int = 100,
) -> dict:
    """
    Create a development mode response for video generation API
    
    Args:
        model: Model name (default: "sora-2")
        prompt: Video generation prompt
        size: Video dimensions (e.g., "1024x1808")
        seconds: Video duration in seconds
        quality: Video quality ("standard" or "hd")
        status: Video job status (default: "completed")
        progress: Generation progress 0-100 (default: 100)
    
    Returns:
        dict: VideoResource compatible response
    """
    
    video_id = f"video_dev_{int(time.time())}"
    
    response_data = {
        "id": video_id,
        "object": "video",
        "model": model,
        "status": status,
        "created_at": int(time.time()),
    }
    
    if status != "completed":
        response_data["progress"] = progress
    
    if size:
        response_data["size"] = size
    if seconds:
        response_data["seconds"] = seconds
    if quality:
        response_data["quality"] = quality
    if prompt:
        response_data["prompt"] = prompt
    
    
    return response_data