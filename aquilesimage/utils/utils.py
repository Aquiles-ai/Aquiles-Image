from fastapi.security import HTTPBearer
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
    api_key: Optional[str] = Security(security)
):
    configs = await load_config_app()

    valid_keys = [k for k in configs["allows_api_keys"] if k and k.strip()]
    
    if not valid_keys:
        return None

    if configs["allows_api_keys"]:
        if not api_key:
            raise HTTPException(
                status_code=403,
                detail="API key missing",
            )
        if api_key not in configs["allows_api_keys"]:
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
