"""
The goal is to create image generation, editing, and variance endpoints compatible with the OpenAI client.

APIs:
POST /images/edits (edit)
POST /images/generations (generate)
"""

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from aquilesimage.models import CreateImageRequest, ImagesResponse, Image, ImageModel, ListModelsResponse, Model, CreateVideoBody, VideoResource, VideoModels, VideoListResource, DeletedVideoResource
from aquilesimage.utils import Utils, setup_colored_logger, verify_api_key, create_dev_mode_response, create_dev_mode_video_response, VideoTaskGeneration, getTypeModel
from aquilesimage.configs import load_config_app, load_config_cli
import asyncio
import logging
from contextlib import asynccontextmanager
import threading
import torch
import os
import gc
import time
import base64
import io
from typing import Optional, Any, List
from datetime import datetime
from aquilesimage.runtime.batch_inf import BatchPipeline
import torch.multiprocessing as mp

DEV_MODE_IMAGE_URL = os.getenv("DEV_IMAGE_URL", "https://picsum.photos/1024/1024")
DEV_MODE_IMAGE_PATH = os.getenv("DEV_IMAGE_PATH", None)

logger = setup_colored_logger("Aquiles-Image", logging.INFO)

logger.info("Loading the model...")

model_pipeline = None
request_pipe = None
pipeline_lock = threading.Lock()
initializer = None
config = None
max_concurrent_infer: int | None = None
load_model: bool | None = None
steps: int | None = None
model_name: str | None = None
video_task_gen: VideoTaskGeneration | None = None
auto_pipeline: bool | None = None ## Because it was a string and because it worked???
device_map_flux2: str | None = None
batch_mode: bool | None = None
batch_pipeline: BatchPipeline | None = None
max_batch_size: int | None = None
batch_timeout: float | None = None
worker_sleep: float | None = None
dist_inference: bool | None = None
Videomodel = [model for model in VideoModels]
worker_manager: Optional[Any] = None

def load_models():
    global model_pipeline, request_pipe, initializer, config, max_concurrent_infer, load_model, steps, model_name, auto_pipeline, device_map_flux2, Videomodel, batch_mode, batch_pipeline, max_batch_size, worker_sleep, batch_timeout, dist_inference

    logger.info("Loading configuration...")
    
    config = load_config_cli() 
    model_name = config.get("model")
    load_model = config.get("load_model")
    auto_pipeline = config.get("auto_pipeline")
    device_map_flux2 = config.get("device_map")
    dist_inference = config.get("dist_inference")
    device_ids = []

    if dist_inference is True:
        logger.info("Distributed inference enabled - configuring multiprocessing...")
        try:
            mp.set_start_method('spawn', force=True)
            logger.info("Multiprocessing start method set to 'spawn'")
        except RuntimeError as e:
            logger.info(f"Multiprocessing already configured: {e}")

    flux_models = [ImageModel.FLUX_1_DEV, ImageModel.FLUX_1_KREA_DEV, ImageModel.FLUX_1_SCHNELL, ImageModel.FLUX_2_4BNB, ImageModel.FLUX_2, ImageModel.FLUX_2_KLEIN_4B, ImageModel.FLUX_2_KLEIN_9B]

    max_concurrent_infer = config.get("max_concurrent_infer")

    steps = config.get("steps_n")

    max_batch_size = config.get("max_batch_size")

    batch_timeout = config.get("batch_timeout")

    worker_sleep = config.get("worker_sleep")

    if steps is not None:
        steps = int(steps)

    if max_concurrent_infer is not None:
        max_concurrent_infer = int(max_concurrent_infer)

    if max_batch_size is not None:
        max_batch_size = int(max_batch_size)

    if batch_timeout is not None:
        batch_timeout = float(batch_timeout)


    if worker_sleep is not None:
        worker_sleep = float(worker_sleep)


    if not model_name:
        raise ValueError("No model specified in configuration. Please configure a model first.")
    
    logger.info(f"Loading model: {model_name}")

    if load_model is False:
        logger.info(f"Dev mode without model loading")
        pass
    else:
        if model_name in  Videomodel:
            try:
                from aquilesimage.pipelines.video import ModelVideoPipelineInit
                initializer = ModelVideoPipelineInit(model_name)
                model_pipeline = initializer.initialize_pipeline()
                model_pipeline.start()
            except Exception as e:
                logger.error(f"Failed to initialize model pipeline: {e}")
                raise
        else:
            if dist_inference is True:
                logger.info("=" * 60)
                logger.info("INITIALIZING DISTRIBUTED INFERENCE")
                logger.info("=" * 60)
                
                try:
                    from aquilesimage.runtime.worker_manager import WorkerManager

                    worker_manager = WorkerManager(
                        model_name=model_name,
                        config=config,
                        num_workers=None
                    )
                    
                    logger.info("Starting workers (this may take a few minutes)...")

                    worker_manager.start()

                    work_queues, result_queues = worker_manager.get_queues()
                    device_ids = worker_manager.get_device_ids()
                    
                    logger.info(f"Workers initialized successfully: {device_ids}")
                    logger.info(f"  - Work queues: {len(work_queues)}")
                    logger.info(f"  - Result queues: {len(result_queues)}")

                    batch_pipeline = BatchPipeline(
                        request_scoped_pipeline=None,
                        work_queues=work_queues,
                        result_queues=result_queues,
                        max_batch_size=max_batch_size if max_batch_size is not None else 4,
                        batch_timeout=batch_timeout if batch_timeout is not None else 0.5,
                        worker_sleep=worker_sleep if worker_sleep is not None else 0.05,
                        is_dist=True,
                        device_ids=device_ids
                    )
                    
                    request_pipe = None
                    model_pipeline = None

                    class DummyInitializer:
                        def __init__(self):
                            self.device = None
                    
                    initializer = DummyInitializer()
                    
                    logger.info("=" * 60)
                    logger.info("DISTRIBUTED INFERENCE READY")
                    logger.info(f"Active workers: {len(device_ids)}")
                    logger.info("=" * 60)
                    
                except Exception as e:
                    logger.error(f"Failed to initialize distributed inference: {e}")
                    raise
            else:
                try:
                    from aquilesimage.runtime import RequestScopedPipeline
                    from aquilesimage.pipelines import ModelPipelineInit
                    if auto_pipeline is True:
                        initializer = ModelPipelineInit(model=model_name, auto_pipeline=True)
                    elif device_map_flux2 == 'cuda' and model_name == ImageModel.FLUX_2_4BNB:
                        initializer = ModelPipelineInit(model=model_name, device_map_flux2='cuda')
                    else:
                        initializer = ModelPipelineInit(model=model_name)

                    model_pipeline = initializer.initialize_pipeline()
                    model_pipeline.start()
        
                    if model_name == ImageModel.FLUX_1_KONTEXT_DEV:
                        if dist_inference is True:
                            request_pipe = RequestScopedPipeline(pipelines=model_pipeline.pipelines, use_kontext=True, is_dist=dist_inference)
                            device_ids = list(model_pipeline.pipelines.keys())
                        else:
                            request_pipe = RequestScopedPipeline(model_pipeline.pipeline, use_kontext=True)
                    elif model_name in flux_models:
                        if dist_inference is True:
                            request_pipe = RequestScopedPipeline(pipelines=model_pipeline.pipelines, use_flux=True, is_dist=dist_inference)
                            device_ids = list(model_pipeline.pipelines.keys())
                        else:
                            request_pipe = RequestScopedPipeline(model_pipeline.pipeline, use_flux=True)
                    else:
                        if dist_inference is True:
                            request_pipe = RequestScopedPipeline(pipelines=model_pipeline.pipelines, is_dist=dist_inference)
                            device_ids = list(model_pipeline.pipelines.keys())
                        else:
                            request_pipe = RequestScopedPipeline(model_pipeline.pipeline)

                    logger.info(f"Model '{model_name}' loaded successfully")

                    batch_pipeline = BatchPipeline(
                        request_scoped_pipeline=request_pipe,
                        max_batch_size=max_batch_size if max_batch_size is not None else 4,
                        batch_timeout=batch_timeout if batch_timeout is not None else 0.5,
                        worker_sleep=worker_sleep if worker_sleep is not None else 0.05,
                        is_dist=dist_inference if dist_inference is True else False,
                        device_ids=device_ids if device_ids is not None or len(device_ids) > 0 else None
                    )

                except Exception as e:
                    logger.error(f"Failed to initialize model pipeline: {e}")
                    raise

class DummyOutput:
    def __init__(self, images):
        self.images = images

try:
    load_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_task_gen, worker_manager

    app.state.total_requests = 0
    app.state.active_inferences = 0
    app.state.metrics_lock = asyncio.Lock()
    app.state.metrics_task = None
    app.state.config = await load_config_app()

    app.state.MODEL_INITIALIZER = initializer
    app.state.MODEL_PIPELINE = model_pipeline
    app.state.REQUEST_PIPE = request_pipe
    app.state.PIPELINE_LOCK = pipeline_lock
    app.state.BATCH_PIPELINE = batch_pipeline
    app.state.WORKER_MANAGER = worker_manager

    app.state.model = app.state.config.get("model")

    app.state.load_model = load_model

    # dumb config
    app.state.utils_app = Utils(
            host="0.0.0.0",
            port=5500,
        )

    if model_name in Videomodel:
        if model_name == "ltx-2":
            video_task_gen = VideoTaskGeneration(
                pipeline=model_pipeline,
                max_concurrent_tasks=1,
                enable_queue=False
            )
        else:
            video_task_gen = VideoTaskGeneration(
                pipeline=model_pipeline.pipeline,
                max_concurrent_tasks=1,
                enable_queue=False
            )
    else:
        video_task_gen = VideoTaskGeneration(
            pipeline=Any,
            max_concurrent_tasks=1,
            enable_queue=False
        )

    await video_task_gen.start()
    if model_name in Videomodel:
        logger.info("Video task manager started")

    if batch_pipeline is not None:
        await batch_pipeline.start()
        logger.info("batch_pipeline started")

    async def metrics_loop():
            try:
                while True:
                    async with app.state.metrics_lock:
                        total = app.state.total_requests
                        active = app.state.active_inferences
                        vram_info = ""
                        if torch.cuda.is_available():
                            try:
                                for i in range(torch.cuda.device_count()):
                                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        
                                    if dist_inference is False or None:
                                        vram_info = f" vram_allocated={allocated:.2f}GB vram_reserved={reserved:.2f}GB vram_total={total_memory:.2f}GB"
                                    else:
                                        vram_info += f" gpu{i}_allocated={allocated:.2f}GB gpu{i}_reserved={reserved:.2f}GB"
                            except Exception as e:
                                logger.error(f"X Error retrieving VRAM information: {e}")
                                vram_info = " vram=error"
                        else:
                            vram_info = " vram=no_gpu"

                    logger.info(f"[METRICS] total_requests={total} active_inferences={active}{vram_info}")
                    if batch_pipeline is not None:
                        logger.info(f"\n [STATS] {batch_pipeline.get_stats_text()}")
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

        if video_task_gen:
            try:
                await video_task_gen.stop()
            except Exception as e:
                logger.warning(f"Error during video_task_gen shutdown: {e}")

        if worker_manager:
            try:
                logger.info("Stopping worker processes...")
                await run_in_threadpool(worker_manager.stop)
                logger.info("Workers stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping workers: {e}")

        if batch_pipeline:
            try:
                await batch_pipeline.stop()
            except Exception as e:
                logger.warning(f"Error during batch_pipeline shutdown: {e}")


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
    if app.state.load_model is False:
        logger.info("[DEV MODE] Generating mock response")
        utils_app = app.state.utils_app
        
        try:
            response_data = create_dev_mode_response(
                DEV_MODE_IMAGE_PATH, 
                DEV_MODE_IMAGE_URL,
                n=input_r.n,
                response_format=input_r.response_format or "url",
                output_format=input_r.output_format or "png",
                size=input_r.size,
                quality=input_r.quality or "auto",
                background=input_r.background or "auto",
                utils_app=utils_app
            )
            images_obj = [Image(**img) for img in response_data["data"]]
            response_data["data"] = images_obj
            
            return ImagesResponse(**response_data)
        
        except Exception as e:
            logger.error(f"X Error in dev mode: {e}")
            raise HTTPException(500, f"Error in dev mode: {e}")

    if app.state.active_inferences >= max_concurrent_infer:
        raise HTTPException(429)
    
    utils_app = app.state.utils_app
    prompt = input_r.prompt
    model = input_r.model
    valid_models = [e.value for e in ImageModel]
    loaded_model = app.state.model
    if model not in valid_models and model != loaded_model:
        raise HTTPException(status_code=503, detail=f"Model not available: {model}")

    if input_r.n is None:
        n = 1
    else:
        n = input_r.n
    size = input_r.size
    response_format = input_r.response_format or "url"
    quality = input_r.quality or "auto"
    background = input_r.background or "auto"
    output_format = input_r.output_format or "png"

    logger.info(f"{prompt[:50]} - num_images_per_prompt: {n}")

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

    try:
        async with app.state.metrics_lock:
            app.state.active_inferences += 1


        if dist_inference:
            device_param = None
        else:
            device_param = initializer.device if initializer else "cuda"

        
        image = await batch_pipeline.submit(
            prompt=prompt,
            height=h,
            width=w,
            num_inference_steps=steps if steps is not None else 30,
            device=device_param,
            timeout=600.0,
            num_images_per_prompt=n,
        )

        if isinstance(image, list):
            output = DummyOutput(image)
        else:
            output = DummyOutput([image])

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
        
    except asyncio.TimeoutError:
        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        logger.error("X Request timed out")
        raise HTTPException(504, "Request timed out")
        
    except Exception as e:
        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        logger.error(f"X Error during inference: {e}")
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
    image: Optional[UploadFile] = File(None, description="Single image to edit (legacy format)"),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]", description="Image(s) to edit (OpenAI/OpenWebUI format)"),
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

    if image is None and image_array is None:
        raise HTTPException(
            status_code=422, 
            detail="X At least one image is required. Use 'image' or 'image[]' parameter."
        )

    images_list: List[UploadFile] = []
    
    if image is not None:
        images_list.append(image)
        logger.info("Received image via 'image' parameter (single)")
    
    if image_array is not None:
        images_list.extend(image_array)
        logger.info(f"Received {len(image_array)} image(s) via 'image[]' parameter (array)")
    
    logger.info(f"Total images to process: {len(images_list)}")

    single_image_models = [ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.QWEN_IMAGE_EDIT_BASE]

    if len(images_list) > 1 and model in single_image_models:
        raise HTTPException(
            status_code=400, 
            detail=f"X Model {model} only supports a single input image. Received {len(images_list)} images."
        )

    if len(images_list) > 10:
        raise HTTPException(
            status_code=400, 
            detail="X Maximum 10 images allowed"
        )

    if model not in [ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.FLUX_2_4BNB, ImageModel.FLUX_2, 
                     ImageModel.QWEN_IMAGE_EDIT_BASE, ImageModel.QWEN_IMAGE_EDIT_2511, ImageModel.QWEN_IMAGE_EDIT_2509, 
                     ImageModel.FLUX_2_KLEIN_4B, ImageModel.FLUX_2_KLEIN_9B, ImageModel.GLM]:
        raise HTTPException(500, f"X Model not available")

    
    if app.state.load_model is False:
        logger.info("[DEV MODE] Generating mock edit response")
        utils_app = app.state.utils_app
        
        try:
            response_data = create_dev_mode_response(
                DEV_MODE_IMAGE_PATH,
                DEV_MODE_IMAGE_URL,
                n=n or 1,
                response_format=response_format or "url",
                output_format=output_format or "png",
                size=size,
                quality=quality or "auto",
                background=background or "auto",
                utils_app=utils_app
            )
            
            images_obj = [Image(**img) for img in response_data["data"]]
            response_data["data"] = images_obj
            
            return ImagesResponse(**response_data)
        
        except Exception as e:
            logger.error(f"X Error in dev mode: {e}")
            raise HTTPException(500, f"X Error in dev mode: {e}")
    
    if app.state.active_inferences >= max_concurrent_infer:
        raise HTTPException(429)

    req_pipe = app.state.REQUEST_PIPE
    utils_app = app.state.utils_app

    if n is None:
        n = 1
    
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

    from PIL import Image as PILImage
    
    try:
        images_pil = []
        for idx, img_file in enumerate(images_list):
            img_content = await img_file.read()
            img_pil = PILImage.open(io.BytesIO(img_content))
            
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            images_pil.append(img_pil)
            logger.info(f"Image {idx+1}/{len(images_list)}: {img_pil.size}, mode: {img_pil.mode}")

        if len(images_pil) == 1:
            image_to_use = images_pil[0]
            logger.info(f"Processing single image: {images_pil[0].size}")
        else:
            image_to_use = images_pil
            logger.info(f"Processing multiple images: {len(images_pil)} images")
            
    except Exception as e:
        raise HTTPException(400, f"X Invalid image file: {str(e)}")

    
    if input_fidelity == "high":
        gd = 5.0
    elif input_fidelity == "low":
        gd = 2.0
    else:
        if model in [ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.FLUX_2_4BNB, ImageModel.FLUX_2,
                ImageModel.FLUX_2_KLEIN_4B, ImageModel.FLUX_2_KLEIN_9B]:
            gd = 3.5 
        else:
            gd = 7.5
    
    try:
        async with app.state.metrics_lock:
            app.state.active_inferences += 1

        if dist_inference:
            device_param = None
        else:
            device_param = initializer.device if initializer else "cuda"


        image_result = await batch_pipeline.submit(
            prompt=prompt,
            image=image_to_use,
            height=h,
            width=w,
            num_inference_steps=steps if steps is not None else 30,
            device=device_param,
            timeout=600.0,
            guidance_scale=gd,
            output_type="pil",
            num_images_per_prompt=n or 1,
            use_glm=True if model in [ ImageModel.GLM ] else False,
        )

        if isinstance(image_result, list):
            output = DummyOutput(image_result)
        else:
            output = DummyOutput([image_result])

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
        logger.error(f"X Error during inference: {e}")
        raise HTTPException(500, f"X Error in processing: {e}")

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


@app.get("/models", response_model=ListModelsResponse, dependencies=[Depends(verify_api_key)], tags=["Models"])
async def get_models():

    type_model = None

    if auto_pipeline is True:
        type_model = "Image"
    else:
        type_model = await getTypeModel(model_name)

    models_data = [
        Model(
            id=f"{model_name} | {type_model}",
            object="model",
            created=int(datetime.now().timestamp()),
            owned_by="custom"
        )
    ]

    return ListModelsResponse(
        object="list",
        data=models_data
    )

@app.post("/videos", response_model=VideoResource, dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def videos(input_r: CreateVideoBody):
    if app.state.load_model is False:
        logger.info("[DEV MODE] Generating mock videos response")
        response = create_dev_mode_video_response(
            model=input_r.model,
            prompt=input_r.prompt,
            size=input_r.size,
            seconds=input_r.seconds,
            quality=input_r.quality,
            status="processing",
            progress=50
        )
        
        return response
    if app.state.model in Videomodel:
        try:
            video_resource = await video_task_gen.create_task(input_r)
            return video_resource
        except Exception as e:
            logger.error(f"X Error creating video task: {e}")
            raise HTTPException(status_code=503, detail=str(e))
    else:
        raise HTTPException(503, f"You are running the model: {app.state.model}. This model does not generate videos.")

@app.get("/videos/{video_id}", response_model=VideoResource, dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def get_video(video_id: str):    
    if app.state.load_model is False:
        return create_dev_mode_video_response(
            model="sora-2",
            prompt="Mock prompt",
            status="completed",
            progress=100
        )
    
    if model_name in Videomodel:
        video = await video_task_gen.get_task(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video
    else:
        raise HTTPException(status_code=503, detail=f"You are running the model: {app.state.model}. This model does not generate videos.")

@app.get("/videos", response_model=VideoListResource, dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def list_videos(
    limit: int = 20,
    after: Optional[str] = None
):
    
    if app.state.load_model is False:
        mock_video = create_dev_mode_video_response(
            model="sora-2",
            prompt="Mock prompt",
            status="completed",
            progress=100
        )
        return VideoListResource(
            data=[mock_video],
            object="list",
            has_more=False,
            first_id=mock_video.id,
            last_id=mock_video.id
        )
    
    if model_name in Videomodel:
        videos, has_more = await video_task_gen.list_tasks(limit, after)
        return VideoListResource(
            data=videos,
            object="list",
            has_more=has_more,
            first_id=videos[0].id if videos else None,
            last_id=videos[-1].id if videos else None
        )
    else:
        raise HTTPException(status_code=503, detail=f"You are running the model: {app.state.model}. This model does not generate videos.")

@app.delete("/videos/{video_id}", response_model=DeletedVideoResource, dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def delete_video(video_id: str):    
    if app.state.load_model is False:
        return DeletedVideoResource(
            id=video_id,
            object="video.deleted",
            deleted=True
        )
    
    if model_name in Videomodel:
        deleted = await video_task_gen.delete_task(video_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Video no encontrado")
    
        return DeletedVideoResource(
            id=video_id,
            object="video.deleted",
            deleted=True
        )
    else:
        raise HTTPException(status_code=503, detail=f"You are running the model: {app.state.model}. This model does not generate videos.")

@app.get("/videos/{video_id}/content", tags=["Video APIs"])
async def get_video(video_id):
    if app.state.load_model is False:
        pass

    if model_name in Videomodel:
        path = await video_task_gen.get_path_video(video_id)

        return FileResponse(path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=503, detail=f"You are running the model: {app.state.model}. This model does not generate videos.")

@app.get("/stats", dependencies=[Depends(verify_api_key)], tags=["Stats APIs"])
async def get_stats():
    if model_name in Videomodel:
        stats = await video_task_gen.get_stats()
        return stats
    else:
        stats = await batch_pipeline.get_stats()
        return stats


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