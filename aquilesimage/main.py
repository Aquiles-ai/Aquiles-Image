"""
The goal is to create image generation, editing, and variance endpoints compatible with the OpenAI client.

APIs:

POST /images/variations (create_variation)
POST /images/edits (edit)
POST /images/generations (generate)
"""

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from aquilesimage.models import CreateImageRequest, ImagesResponse, Image, ImageModel
from aquilesimage.utils import Utils, setup_colored_logger, verify_api_key
from aquilesimage.runtime import RequestScopedPipeline
from aquilesimage.pipelines import ModelPipelineInit
from aquilesimage.configs import load_config_app, load_config_cli
import asyncio
import logging
from contextlib import asynccontextmanager
import threading
import torch
import random
import os
import gc
import time
import base64
import io
from typing import Optional

logger = setup_colored_logger("Aquiles-Image", logging.INFO)

logger.info("Loading the model...")

model_pipeline = None
request_pipe = None
pipeline_lock = threading.Lock()
initializer = None
config = None
max_concurrent_infer: int | None = None

def load_models():
    global model_pipeline, request_pipe, initializer, config, max_concurrent_infer

    logger.info("Loading configuration...")
    
    config = load_config_cli() 
    model_name = config.get("model")

    flux_models = [ImageModel.FLUX_1_DEV, ImageModel.FLUX_1_KREA_DEV, ImageModel.FLUX_1_SCHNELL]

    max_concurrent_infer = int(config.get("max_concurrent_infer"))

    if not model_name:
        raise ValueError("No model specified in configuration. Please configure a model first.")
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        initializer = ModelPipelineInit(model=model_name)
        model_pipeline = initializer.initialize_pipeline()
        model_pipeline.start()
        
        if model_name in flux_models:
            request_pipe = RequestScopedPipeline(model_pipeline.pipeline, use_flux=True)
        request_pipe = RequestScopedPipeline(model_pipeline.pipeline)
        
        logger.info(f"Model '{model_name}' loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model pipeline: {e}")
        raise

try:
    load_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.total_requests = 0
    app.state.active_inferences = 0
    app.state.metrics_lock = asyncio.Lock()
    app.state.metrics_task = None
    app.state.config = await load_config_app()

    app.state.MODEL_INITIALIZER = initializer
    app.state.MODEL_PIPELINE = model_pipeline
    app.state.REQUEST_PIPE = request_pipe
    app.state.PIPELINE_LOCK = pipeline_lock

    app.state.model = app.state.config.get("model")

    # dumb config
    app.state.utils_app = Utils(
            host="0.0.0.0",
            port=5500,
        )

    async def metrics_loop():
            try:
                while True:
                    async with app.state.metrics_lock:
                        total = app.state.total_requests
                        active = app.state.active_inferences
                    logger.info(f"[METRICS] total_requests={total} active_inferences={active}")
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                logger.info("Metrics loop cancelled")
                raise

    app.state.metrics_task = asyncio.create_task(metrics_loop())

    try:
        yield
    finally:
        task = app.state.metrics_task
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            stop_fn = getattr(model_pipeline, "stop", None) or getattr(model_pipeline, "close", None)
            if callable(stop_fn):
                await run_in_threadpool(stop_fn)
        except Exception as e:
            logger.warning(f"Error during pipeline shutdown: {e}")

        if model_pipeline:
            try:
                stop_fn = getattr(model_pipeline, "stop", None) or getattr(model_pipeline, "close", None)
                if callable(stop_fn):
                    await run_in_threadpool(stop_fn)
                    logger.info("Model pipeline stopped successfully")
            except Exception as e:
                logger.warning(f"Error during pipeline shutdown: {e}")

        logger.info("Lifespan shutdown complete")

app = FastAPI(title="Aquiles-Image", lifespan=lifespan)

@app.middleware("http")
async def count_requests_middleware(request: Request, call_next):
    async with app.state.metrics_lock:
        app.state.total_requests += 1
    response = await call_next(request)
    return response

@app.post("/images/generations", response_model=ImagesResponse, tags=["Generation"], dependencies=[Depends(verify_api_key)])
async def create_image(input_r: CreateImageRequest):
    if app.state.active_inferences >= max_concurrent_infer:
        raise HTTPException(429)
    
    utils_app = app.state.utils_app

    def make_generator():
        g = torch.Generator(device=initializer.device)
        return g.manual_seed(random.randint(0, 10_000_000))

    prompt = input_r.prompt
    model = input_r.model

    if model not in [e.value for e in ImageModel] or model not in app.state.model:
        HTTPException(500, f"Model not available")

    n = input_r.n
    size = input_r.size
    response_format = input_r.response_format or "url"
    quality = input_r.quality or "auto"
    background = input_r.background or "auto"
    output_format = input_r.output_format or "png"

    if size == "1024x1024":
        h, w = 1024, 1024
    elif size == "1536x1024":
        h, w = 1536, 1024
    elif size == "1024x1536":
        h, w = 1024, 1536
    elif size == "256x256":
        h, w = 256, 256
    elif size == "512x512":
        h, w = 512, 512
    elif size == "1792x1024":
        h, w = 1792, 1024
    elif size == "1024x1792":
        h, w = 1024, 1792
    else:
        h, w = 1024, 1024
        size = "1024x1024"

    req_pipe = app.state.REQUEST_PIPE

    def infer():
        gen = make_generator()
        return req_pipe.generate(
            prompt=prompt,
            generator=gen,
            num_inference_steps=30,
            height=h,
            width=w,
            num_images_per_prompt=n,
            device=initializer.device,
            output_type="pil",
        )

    try:
        async with app.state.metrics_lock:
            app.state.active_inferences += 1

        output = await run_in_threadpool(infer)

        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        
        images_data = []
        
        for img in output.images:
            image_obj = {}
            
            if response_format == "b64_json":
                buffer = io.BytesIO()
                img.save(buffer, format=output_format.upper())
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_obj["b64_json"] = img_str
            else:
                url = utils_app.save_image(img)
                image_obj["url"] = url
            
            images_data.append(Image(**image_obj))

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
            

        return ImagesResponse(**response_data)
        
    except Exception as e:
        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        logger.error(f"Error during inference: {e}")
        raise HTTPException(500, f"Error in processing: {e}")

    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
        gc.collect()


@app.post("/images/edits", response_model=ImagesResponse, tags=["Edit"], dependencies=[Depends(verify_api_key)])  
async def create_image_edit(
    image: UploadFile = File(..., description="The image to edit"),
    mask: Optional[UploadFile] = File(None, description="An additional image to be used as a mask"),
    prompt: str = Form(..., max_length=1000, description="A text description of the desired image(s)."),
    background: Optional[str] = Form(None, description="Allows to set transparency for the background"),
    model: Optional[str] = Form(None, description="The model to use for image generation"),
    n: Optional[int] = Form(1, ge=1, le=10, description="The number of images to generate"),
    size: Optional[str] = Form("1024x1024", description="The size of the generated images"),
    response_format: Optional[str] = Form("url", description="The format in which the generated images are returned"),
    output_format: Optional[str] = Form("png", description="The format in which the generated images are returned"),
    output_compression: Optional[int] = Form(None, description="The compression level for the generated images"),
    user: Optional[str] = Form(None, description="A unique identifier representing your end-user"),
    input_fidelity: Optional[str] = Form(None, description="Control how much effort the model will exert"),
    stream: Optional[bool] = Form(False, description="Edit the image in streaming mode"),
    partial_images: Optional[int] = Form(None, ge=0, le=3, description="The number of partial images to generate"),
    quality: Optional[str] = Form("auto", description="The quality of the image that will be generated")
):
    if app.state.active_inferences >= max_concurrent_infer:
        raise HTTPException(429)

    def make_generator():
        g = torch.Generator(device=initializer.device)
        return g.manual_seed(random.randint(0, 10_000_000))

    req_pipe = app.state.REQUEST_PIPE
    utils_app = app.state.utils_app

    if model not in [ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.QWEN_IMAGE_EDIT] or model not in app.state.model:
        raise HTTPException(500, f"Model not available")

    try:
        image_content = await image.read()

        from PIL import Image as PILImage
        image_pil = PILImage.open(io.BytesIO(image_content))

        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        width, height = image_pil.size
        logger.info(f"Original image: {width}x{height}, mode: {image_pil.mode}")
        
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {str(e)}")

    if size == "1024x1024":
        h, w = 1024, 1024
    elif size == "1536x1024":
        h, w = 1536, 1024
    elif size == "1024x1536":
        h, w = 1024, 1536
    elif size == "256x256":
        h, w = 256, 256
    elif size == "512x512":
        h, w = 512, 512
    elif size == "1792x1024":
        h, w = 1792, 1024
    elif size == "1024x1792":
        h, w = 1024, 1792
    else:
        h, w = 1024, 1024
        size = "1024x1024"

    image_to_use = image_pil  
    
    if model == ImageModel.FLUX_1_KONTEXT_DEV:
        logger.info(f"Flux Kontext: Using original image without resizing: {image_pil.size}")
        image_to_use = image_pil
        
    elif model == ImageModel.QWEN_IMAGE_EDIT:
        if image_pil.size != (w, h):
            logger.info(f"QwenImageEdit: Resizing image from {image_pil.size} to {(w, h)}")
            image_to_use = image_pil.resize((w, h), PILImage.Resampling.LANCZOS)
        else:
            logger.info(f"QwenImageEdit: Image is now the correct size: {(w, h)}")
            image_to_use = image_pil

    if input_fidelity == "high":
        gd = 5.0
    elif input_fidelity == "low":
        gd = 2.0
    else:
        if model == ImageModel.FLUX_1_KONTEXT_DEV:
            gd = 2.5 
        else:
            gd = 7.5  

    def infer():
        gen = make_generator()
        
        if model == ImageModel.FLUX_1_KONTEXT_DEV:
            logger.info(f"FluxKontext inference - guidance_scale: {gd}")
            return req_pipe.generate(
                image=image_to_use,
                prompt=prompt,
                guidance_scale=gd,
                generator=gen, 
                num_images_per_prompt=n or 1,  
                output_type="pil",  
            )
        else:
            logger.info(f"QwenImageEdit inference - guidance_scale: {gd}, size: {h}x{w}")
            return req_pipe.generate(
                image=image_to_use,
                prompt=prompt,
                generator=gen,
                guidance_scale=gd,
                num_inference_steps=30,
                height=h,
                width=w,
                num_images_per_prompt=n or 1,
                output_type="pil",
            )

    try:
        async with app.state.metrics_lock:
            app.state.active_inferences += 1

        output = await run_in_threadpool(infer)

        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        
        images_data = []
        
        for img in output.images:
            image_obj = {}
            
            if response_format == "b64_json":
                buffer = io.BytesIO()
                img.save(buffer, format=output_format.upper())
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_obj["b64_json"] = img_str
            else:
                url = utils_app.save_image(img)
                image_obj["url"] = url
            
            images_data.append(Image(**image_obj))

        response_data = {
            "created": int(time.time()),
            "data": images_data,
        }
        
        if model != ImageModel.FLUX_1_KONTEXT_DEV and size:
            response_data["size"] = size
        if quality:
            response_data["quality"] = quality
        if background:
            response_data["background"] = background
        if output_format:
            response_data["output_format"] = output_format

        return ImagesResponse(**response_data)
        
    except Exception as e:
        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        logger.error(f"Error during inference: {e}")
        raise HTTPException(500, f"Error in processing: {e}")

    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
        gc.collect()


@app.get("/images/{filename}", tags=["Download Images"])
async def serve_image(filename: str):
    utils_app = app.state.utils_app
    file_path = os.path.join(utils_app.image_dir, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/png")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5500)