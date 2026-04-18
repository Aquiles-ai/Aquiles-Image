"""
Image generation, editing, and variance endpoints compatible with the OpenAI client.

APIs:
POST /images/edits    (edit)
POST /images/generations (generate)
"""

from __future__ import annotations
import asyncio
import base64
import gc
import io
import logging
import os
import pathlib
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, List, Literal, Optional

import torch
import torch.multiprocessing as mp
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage

from aquilesimage.auth import (ACCESS_TOKEN_EXPIRE_MINUTES, authenticate_user,
                                create_access_token, get_current_user)
from aquilesimage.configs import load_config_app, load_config_cli
from aquilesimage.models import (CreateImageRequest, CreateVideoBody,
                                  DeletedVideoResource, Image, ImageModel,
                                  ImagesResponse, ListModelsResponse, Model,
                                  VideoListResource, VideoModels, VideoQuality,
                                  VideoResource)
from aquilesimage.runtime.batch_inf import BatchPipeline
from aquilesimage.utils import (Utils, VideoTaskGeneration, create_dev_mode_response,
                                 create_dev_mode_video_response, getTypeModel,
                                 setup_colored_logger, verify_api_key)


DEV_MODE_IMAGE_URL = os.getenv("DEV_IMAGE_URL", "https://picsum.photos/1024/1024")
DEV_MODE_IMAGE_PATH = os.getenv("DEV_IMAGE_PATH", None)

SIZE_MAP: dict[str, tuple[int, int]] = {
    "1024x1024": (1024, 1024),
    "1536x1024": (1536, 1024),
    "1024x1536": (1024, 1536),
    "256x256":   (256,  256),
    "512x512":   (512,  512),
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
}

FLUX_MODELS = frozenset([
    ImageModel.FLUX_1_DEV, ImageModel.FLUX_1_KREA_DEV, ImageModel.FLUX_1_SCHNELL,
    ImageModel.FLUX_2_4BNB, ImageModel.FLUX_2, ImageModel.FLUX_2_KLEIN_4B,
    ImageModel.FLUX_2_KLEIN_9B, ImageModel.FLUX_2_KLEIN_9B_KV,
])

EDIT_SUPPORTED_MODELS = frozenset([
    ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.FLUX_2_4BNB, ImageModel.FLUX_2,
    ImageModel.QWEN_IMAGE_EDIT_BASE, ImageModel.QWEN_IMAGE_EDIT_2511,
    ImageModel.QWEN_IMAGE_EDIT_2509, ImageModel.FLUX_2_KLEIN_4B,
    ImageModel.FLUX_2_KLEIN_9B, ImageModel.GLM, ImageModel.FLUX_2_KLEIN_9B_KV,
])

VIDEO_MODELS = frozenset(VideoModels)

logger = setup_colored_logger("Aquiles-Image", logging.INFO)


@dataclass
class AppConfig:
    model_name: str = ""
    load_model: Optional[bool] = None
    auto_pipeline: Optional[bool] = None
    auto_type: Optional[str] = None
    device_map_flux2: Optional[str] = None
    dist_inference: Optional[bool] = None
    max_concurrent_infer: int = 4
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    load_lora: bool = False
    lora_config_path: Optional[str] = None
    allow_users: bool = False
    max_batch_size: int = 4
    batch_timeout: float = 0.5
    worker_sleep: float = 0.05


cfg = AppConfig()
model_pipeline = None
request_pipe = None
initializer = None
batch_pipeline: Optional[BatchPipeline] = None
worker_manager: Optional[Any] = None
pipeline_lock = threading.Lock()
video_task_gen: Optional[VideoTaskGeneration] = None

def parse_size(size: Optional[str]) -> tuple[int, int, str]:
    h, w = SIZE_MAP.get(size or "", (1024, 1024))
    canonical = size if size in SIZE_MAP else "1024x1024"
    return h, w, canonical

class DummyOutput:
    def __init__(self, images):
        self.images = images if isinstance(images, list) else [images]

def build_images_response(
    output: DummyOutput,
    utils_app: Utils,
    response_format: str,
    output_format: str,
    size: Optional[str],
    quality: Optional[str],
    background: Optional[str],
    skip_size: bool = False,
) -> ImagesResponse:
    images_data = []
    for img in output.images:
        if response_format == "b64_json":
            buf = io.BytesIO()
            img.save(buf, format=output_format.upper())
            images_data.append(Image(b64_json=base64.b64encode(buf.getvalue()).decode()))
        else:
            images_data.append(Image(url=utils_app.save_image(img)))

    payload: dict[str, Any] = {"created": int(time.time()), "data": images_data}
    if not skip_size and size:
        payload["size"] = size
    if quality:
        payload["quality"] = quality
    if background:
        payload["background"] = background
    if output_format:
        payload["output_format"] = output_format
    return ImagesResponse(**payload)


def _load_lora_conf(cfg: AppConfig):
    if cfg.load_lora and cfg.lora_config_path:
        from aquilesimage.configs import load_lora_config
        conf = load_lora_config(cfg.lora_config_path)
        if conf is None:
            logger.warning("LoRA config could not be loaded. Continuing without LoRA.")
            cfg.load_lora = False
            return None
        return conf
    return None


def _init_batch_pipeline(request_scoped_pipe, device_ids, cfg: AppConfig, work_queues=None, result_queues=None, is_dist=False) -> BatchPipeline:
    return BatchPipeline(
        request_scoped_pipeline=request_scoped_pipe,
        work_queues=work_queues,
        result_queues=result_queues,
        max_batch_size=cfg.max_batch_size,
        batch_timeout=cfg.batch_timeout,
        worker_sleep=cfg.worker_sleep,
        is_dist=is_dist,
        device_ids=device_ids or None,
    )


def _load_video_pipeline(cfg: AppConfig):
    from aquilesimage.pipelines.video import ModelVideoPipelineInit
    init = ModelVideoPipelineInit(cfg.model_name)
    pipeline = init.initialize_pipeline()
    pipeline.start()
    return pipeline, None, None


def _load_distributed_pipeline(cfg: AppConfig):
    from aquilesimage.runtime.worker_manager import WorkerManager

    logger.info("Initializing distributed inference...")
    mp.set_start_method("spawn", force=True)

    wm = WorkerManager(model_name=cfg.model_name, config=vars(cfg), num_workers=None)
    wm.start()
    work_q, result_q = wm.get_queues()
    device_ids = wm.get_device_ids()

    class _DummyInit:
        device = None

    bp = _init_batch_pipeline(None, device_ids, cfg, work_q, result_q, is_dist=True)
    logger.info(f"Distributed inference ready — workers: {device_ids}")
    return None, None, _DummyInit(), bp, wm, device_ids


def _load_single_pipeline(cfg: AppConfig, conf_lora):
    from aquilesimage.pipelines import ModelPipelineInit
    from aquilesimage.runtime import RequestScopedPipeline

    kwargs = dict(load_lora=cfg.load_lora, conf_lora=conf_lora)
    if cfg.auto_pipeline:
        init = ModelPipelineInit(model=cfg.model_name, auto_pipeline=True, auto_type=cfg.auto_type, **kwargs)
    elif cfg.device_map_flux2 == "cuda" and cfg.model_name == ImageModel.FLUX_2_4BNB:
        init = ModelPipelineInit(model=cfg.model_name, device_map_flux2="cuda", **kwargs)
    else:
        init = ModelPipelineInit(model=cfg.model_name, **kwargs)

    pipeline = init.initialize_pipeline()
    pipeline.start()

    use_kontext = cfg.model_name == ImageModel.FLUX_1_KONTEXT_DEV
    use_flux = cfg.model_name in FLUX_MODELS
    rp = RequestScopedPipeline(pipeline.pipeline, use_kontext=use_kontext, use_flux=use_flux)

    bp = _init_batch_pipeline(rp, [], cfg)
    logger.info(f"Model '{cfg.model_name}' loaded successfully")
    return pipeline, rp, init, bp


def load_models():
    global model_pipeline, request_pipe, initializer, batch_pipeline, worker_manager, cfg

    raw = load_config_cli(use_cache=False)

    cfg.model_name       = raw.get("model") or ""
    cfg.load_model       = raw.get("load_model")
    cfg.auto_pipeline    = raw.get("auto_pipeline")
    cfg.device_map_flux2 = raw.get("device_map")
    cfg.dist_inference   = raw.get("dist_inference")
    cfg.auto_type        = raw.get("auto_pipeline_mode")
    cfg.guidance_scale   = raw.get("guidance_scale")
    cfg.seed             = raw.get("seed")
    cfg.load_lora        = raw.get("load_lora") or False
    cfg.lora_config_path = raw.get("lora_config_path")
    cfg.steps            = int(raw["steps_n"]) if raw.get("steps_n") else None
    cfg.max_concurrent_infer = int(raw["max_concurrent_infer"]) if raw.get("max_concurrent_infer") else 4
    cfg.max_batch_size   = int(raw["max_batch_size"]) if raw.get("max_batch_size") else 4
    cfg.batch_timeout    = float(raw["batch_timeout"]) if raw.get("batch_timeout") else 0.5
    cfg.worker_sleep     = float(raw["worker_sleep"]) if raw.get("worker_sleep") else 0.05

    allows = raw.get("allows_users") or []
    cfg.allow_users = bool(allows)
    logger.info(f"{'There are' if cfg.allow_users else 'No'} users{f': {len(allows)}' if cfg.allow_users else ''}")

    if not cfg.model_name:
        raise ValueError("No model specified in configuration. Please configure a model first.")

    logger.info(f"Loading model: {cfg.model_name}")

    if cfg.load_model is False:
        logger.info("Dev mode — model loading skipped")
        return

    conf_lora = _load_lora_conf(cfg)

    if cfg.model_name in VIDEO_MODELS:
        model_pipeline, request_pipe, initializer = _load_video_pipeline(cfg)
    elif cfg.dist_inference:
        model_pipeline, request_pipe, initializer, batch_pipeline, worker_manager, _ = _load_distributed_pipeline(cfg)
    else:
        model_pipeline, request_pipe, initializer, batch_pipeline = _load_single_pipeline(cfg, conf_lora)


logger.info("Loading the model...")
try:
    load_models()
except Exception as exc:
    logger.error(f"X Failed to initialize models: {exc}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_task_gen

    app.state.total_requests = 0
    app.state.active_inferences = 0
    app.state.metrics_lock = asyncio.Lock()
    app.state.config = await load_config_app()
    app.state.MODEL_INITIALIZER = initializer
    app.state.MODEL_PIPELINE = model_pipeline
    app.state.REQUEST_PIPE = request_pipe
    app.state.PIPELINE_LOCK = pipeline_lock
    app.state.BATCH_PIPELINE = batch_pipeline
    app.state.WORKER_MANAGER = worker_manager
    app.state.model = cfg.model_name
    app.state.load_model = cfg.load_model
    app.state.utils_app = Utils(host="0.0.0.0", port=5500)

    vt_pipeline = (
        model_pipeline if cfg.model_name in ("ltx-2", "ltx-2.3")
        else getattr(model_pipeline, "pipeline", Any)
    ) if cfg.model_name in VIDEO_MODELS else Any

    video_task_gen = VideoTaskGeneration(pipeline=vt_pipeline, max_concurrent_tasks=1, enable_queue=False)
    await video_task_gen.start()

    if batch_pipeline is not None:
        await batch_pipeline.start()
        logger.info("batch_pipeline started")

    async def metrics_loop():
        try:
            while True:
                async with app.state.metrics_lock:
                    total, active = app.state.total_requests, app.state.active_inferences
                vram = ""
                if torch.cuda.is_available():
                    try:
                        for i in range(torch.cuda.device_count()):
                            a = torch.cuda.memory_allocated(i) / 1024**3
                            r = torch.cuda.memory_reserved(i) / 1024**3
                            t = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            vram += (f" vram_allocated={a:.2f}GB vram_reserved={r:.2f}GB vram_total={t:.2f}GB"
                                     if not cfg.dist_inference
                                     else f" gpu{i}_allocated={a:.2f}GB gpu{i}_reserved={r:.2f}GB")
                    except Exception as e:
                        logger.error(f"X Error retrieving VRAM: {e}")
                        vram = " vram=error"
                else:
                    vram = " vram=no_gpu"

                logger.info(f"[METRICS] total_requests={total} active_inferences={active}{vram}")
                if batch_pipeline is not None:
                    logger.info(f"\n [STATS] {batch_pipeline.get_stats_text()}")
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise

    app.state.metrics_task = asyncio.create_task(metrics_loop())

    try:
        yield
    finally:
        app.state.metrics_task.cancel()
        try:
            await app.state.metrics_task
        except asyncio.CancelledError:
            pass

        for obj, attr in [(model_pipeline, "stop"), (model_pipeline, "close")]:
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    await run_in_threadpool(fn)
                except Exception as e:
                    logger.warning(f"Pipeline shutdown error: {e}")
                break

        for task, coro in [
            (video_task_gen, lambda: video_task_gen.stop()),
            (batch_pipeline,  lambda: batch_pipeline.stop()),
        ]:
            if task:
                try:
                    await coro()
                except Exception as e:
                    logger.warning(f"Shutdown error: {e}")

        if worker_manager:
            try:
                await run_in_threadpool(worker_manager.stop)
            except Exception as e:
                logger.warning(f"Worker stop error: {e}")

        logger.info("Lifespan shutdown complete")


package_dir  = pathlib.Path(__file__).parent.absolute()
templates    = Jinja2Templates(directory=os.path.join(package_dir, "templates"))

app = FastAPI(title="Aquiles-Image", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(package_dir, "static")), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def count_requests_middleware(request: Request, call_next):
    async with app.state.metrics_lock:
        app.state.total_requests += 1
    return await call_next(request)

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()
    gc.collect()

@app.post("/images/generations", response_model=ImagesResponse, tags=["Generation"],
          dependencies=[Depends(verify_api_key)])
async def create_image(input_r: CreateImageRequest):
    utils_app = app.state.utils_app
    response_format = input_r.response_format or "url"
    output_format   = input_r.output_format   or "png"
    quality         = input_r.quality         or "auto"
    background      = input_r.background      or "auto"
    n               = input_r.n or 1
    h, w, size      = parse_size(input_r.size)

    if app.state.load_model is False:
        logger.info("[DEV MODE] Generating mock response")
        data = create_dev_mode_response(DEV_MODE_IMAGE_PATH, DEV_MODE_IMAGE_URL,
                                        n=n, response_format=response_format,
                                        output_format=output_format, size=size,
                                        quality=quality, background=background,
                                        utils_app=utils_app)
        data["data"] = [Image(**i) for i in data["data"]]
        return ImagesResponse(**data)

    if app.state.active_inferences >= cfg.max_concurrent_infer:
        raise HTTPException(429)

    valid_models = {e.value for e in ImageModel} | {app.state.model}
    if input_r.model not in valid_models:
        raise HTTPException(503, f"Model not available: {input_r.model}")

    logger.info(f"{input_r.prompt[:50]} - num_images_per_prompt: {n}")

    submit_kwargs: dict[str, Any] = dict(
        prompt=input_r.prompt,
        height=h, width=w,
        num_inference_steps=cfg.steps or 30,
        device=None if cfg.dist_inference else (initializer.device if initializer else "cuda"),
        timeout=600.0,
        num_images_per_prompt=n,
        seed=cfg.seed,
    )
    if input_r.model not in [ImageModel.FLUX_2_KLEIN_9B_KV]:
        submit_kwargs["guidance_scale"] = cfg.guidance_scale or 4

    try:
        async with app.state.metrics_lock:
            app.state.active_inferences += 1
        images = await batch_pipeline.submit(**submit_kwargs)
        return build_images_response(
            DummyOutput(images), utils_app, response_format, output_format, size, quality, background
        )
    except asyncio.TimeoutError:
        logger.error("X Request timed out")
        raise HTTPException(504, "Request timed out")
    except Exception as e:
        logger.error(f"X Error during inference: {e}")
        raise HTTPException(500, f"Error in processing: {e}")
    finally:
        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        _cleanup_gpu()



@app.post("/images/edits", response_model=ImagesResponse, tags=["Edit"],
          dependencies=[Depends(verify_api_key)])
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
    files = ([image] if image else []) + (image_array or [])
    if not files:
        raise HTTPException(422, "At least one image is required.")
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 images allowed.")
    if len(files) > 1 and model in [ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.QWEN_IMAGE_EDIT_BASE]:
        raise HTTPException(400, f"Model {model} only supports a single input image.")

    edit_models = EDIT_SUPPORTED_MODELS | ({cfg.model_name} if cfg.auto_pipeline and cfg.auto_type == "i2i" else set())
    if model not in edit_models:
        raise HTTPException(500, "Model not available")

    utils_app       = app.state.utils_app
    response_format = response_format or "url"
    output_format   = output_format   or "png"
    quality         = quality         or "auto"
    n               = n or 1
    h, w, size      = parse_size(size)

    if app.state.load_model is False:
        logger.info("[DEV MODE] Generating mock edit response")
        data = create_dev_mode_response(DEV_MODE_IMAGE_PATH, DEV_MODE_IMAGE_URL,
                                        n=n, response_format=response_format,
                                        output_format=output_format, size=size,
                                        quality=quality, background=background or "auto",
                                        utils_app=utils_app)
        data["data"] = [Image(**i) for i in data["data"]]
        return ImagesResponse(**data)

    if app.state.active_inferences >= cfg.max_concurrent_infer:
        raise HTTPException(429)

    try:
        images_pil = []
        for idx, f in enumerate(files):
            content = await f.read()
            img = PILImage.open(io.BytesIO(content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images_pil.append(img)
            logger.info(f"Image {idx+1}/{len(files)}: {img.size}, mode: {img.mode}")
        image_to_use = images_pil[0] if len(images_pil) == 1 else images_pil
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {e}")

    if input_fidelity == "high":
        gd = 5.0
    elif input_fidelity == "low":
        gd = 2.0
    else:
        gd = 3.5 if model in [ImageModel.FLUX_1_KONTEXT_DEV, ImageModel.FLUX_2_4BNB,
                               ImageModel.FLUX_2, ImageModel.FLUX_2_KLEIN_4B,
                               ImageModel.FLUX_2_KLEIN_9B] else 7.5

    submit_kwargs: dict[str, Any] = dict(
        prompt=prompt,
        image=image_to_use,
        height=h, width=w,
        num_inference_steps=cfg.steps or 30,
        device=None if cfg.dist_inference else (initializer.device if initializer else "cuda"),
        timeout=600.0,
        output_type="pil",
        num_images_per_prompt=n,
        use_glm=model == ImageModel.GLM,
        seed=cfg.seed,
    )
    if model not in [ImageModel.FLUX_2_KLEIN_9B_KV]:
        submit_kwargs["guidance_scale"] = cfg.guidance_scale or gd

    skip_size = model == ImageModel.FLUX_1_KONTEXT_DEV

    try:
        async with app.state.metrics_lock:
            app.state.active_inferences += 1
        result = await batch_pipeline.submit(**submit_kwargs)
        return build_images_response(
            DummyOutput(result), utils_app, response_format, output_format,
            size, quality, background, skip_size=skip_size
        )
    except Exception as e:
        logger.error(f"X Error during inference: {e}")
        raise HTTPException(500, f"Error in processing: {e}")
    finally:
        async with app.state.metrics_lock:
            app.state.active_inferences = max(0, app.state.active_inferences - 1)
        _cleanup_gpu()



@app.get("/images/{filename}", tags=["Download Images"])
async def serve_image(filename: str):
    path = os.path.join(app.state.utils_app.image_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/models", response_model=ListModelsResponse,
         dependencies=[Depends(verify_api_key)], tags=["Models"])
async def get_models():
    if cfg.auto_pipeline:
        type_model = "Image" if cfg.auto_type == "t2i" else "Edit"
    else:
        type_model = await getTypeModel(cfg.model_name)
    return ListModelsResponse(
        object="list",
        data=[Model(id=f"{cfg.model_name} | {type_model}", object="model",
                    created=int(datetime.now().timestamp()), owned_by="custom")]
    )


def _require_video_model():
    if cfg.model_name not in VIDEO_MODELS:
        raise HTTPException(503, f"Model '{cfg.model_name}' does not generate videos.")


@app.post("/videos", response_model=VideoResource,
          dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def create_video(request: Request):
    content_type = request.headers.get("content-type", "")
    MODELS_WITH_IMAGE = [VideoModels.LTX_2]
    pil_image = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        input_r = CreateVideoBody(
            model=form.get("model", "sora-2"), prompt=form.get("prompt"),
            size=form.get("size"), seconds=form.get("seconds"),
            quality=form.get("quality", VideoQuality.standard),
        )
        ref = form.get("input_reference")
        if ref is not None:
            if input_r.model not in MODELS_WITH_IMAGE:
                raise HTTPException(400, f"'{input_r.model}' does not support input_reference.")
            if ref.content_type not in ("image/jpeg", "image/png", "image/webp"):
                raise HTTPException(400, "Unsupported image format.")
            pil_image = PILImage.open(io.BytesIO(await ref.read()))
    else:
        input_r = CreateVideoBody(**(await request.json()))

    if app.state.load_model is False:
        return create_dev_mode_video_response(model=input_r.model, prompt=input_r.prompt,
                                              size=input_r.size, seconds=input_r.seconds,
                                              quality=input_r.quality, status="processing", progress=50)
    _require_video_model()
    try:
        return await video_task_gen.create_task(input_r, pil_image)
    except Exception as e:
        logger.error(f"X Error creating video task: {e}")
        raise HTTPException(503, str(e))


@app.get("/videos/{video_id}", response_model=VideoResource,
         dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def get_video(video_id: str):
    if app.state.load_model is False:
        return create_dev_mode_video_response(model="sora-2", prompt="Mock", status="completed", progress=100)
    _require_video_model()
    video = await video_task_gen.get_task(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return video


@app.get("/videos", response_model=VideoListResource,
         dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def list_videos(limit: int = 20, after: Optional[str] = None):
    if app.state.load_model is False:
        mock = create_dev_mode_video_response(model="sora-2", prompt="Mock", status="completed", progress=100)
        return VideoListResource(data=[mock], object="list", has_more=False,
                                 first_id=mock.id, last_id=mock.id)
    _require_video_model()
    videos, has_more = await video_task_gen.list_tasks(limit, after)
    return VideoListResource(data=videos, object="list", has_more=has_more,
                             first_id=videos[0].id if videos else None,
                             last_id=videos[-1].id if videos else None)


@app.delete("/videos/{video_id}", response_model=DeletedVideoResource,
            dependencies=[Depends(verify_api_key)], tags=["Video APIs"])
async def delete_video(video_id: str):
    if app.state.load_model is False:
        return DeletedVideoResource(id=video_id, object="video.deleted", deleted=True)
    _require_video_model()
    if not await video_task_gen.delete_task(video_id):
        raise HTTPException(404, "Video not found")
    return DeletedVideoResource(id=video_id, object="video.deleted", deleted=True)


@app.get("/videos/{video_id}/content", tags=["Video APIs"])
async def get_video_content(video_id: str):
    _require_video_model()
    path = await video_task_gen.get_path_video(video_id)
    return FileResponse(path, media_type="video/mp4")


@app.get("/stats", dependencies=[Depends(verify_api_key)], tags=["Stats APIs"])
async def get_stats():
    if app.state.load_model is False:
        return {"mode": "single-device", "total_requests": 150, "total_batches": 42,
                "total_images": 180, "queued": 3, "completed": 147, "failed": 0,
                "processing": True, "available": False}
    if cfg.model_name in VIDEO_MODELS:
        return await video_task_gen.get_stats()
    return await batch_pipeline.get_stats()


# Playground

if cfg.allow_users:
    @app.exception_handler(HTTPException)
    async def auth_exception_handler(request: Request, exc: HTTPException):
        if exc.status_code == 401:
            return RedirectResponse(url=f"/login?next={request.url.path}", status_code=302)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.post("/token", tags=["HTML Helper"])
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        if not await authenticate_user(form_data.username, form_data.password):
            raise HTTPException(401, "Invalid username or password")
        token = await create_access_token(form_data.username,
                                          timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        resp = RedirectResponse(url="/", status_code=302)
        resp.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
        return resp

    @app.get("/login", response_class=HTMLResponse, tags=["HTML"])
    async def login(request: Request):
        return templates.TemplateResponse(request=request, name="login.html")

    @app.get("/", response_class=HTMLResponse, tags=["HTML"])
    async def home(request: Request, user: str = Depends(get_current_user)):
        config_data = app.state.config
        keys = [k for k in config_data.get("allows_api_keys", []) if k and k.strip()]
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={"api_key": keys[0] if keys else "EMPTY"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5500)