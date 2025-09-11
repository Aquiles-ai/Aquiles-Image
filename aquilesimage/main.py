"""
The goal is to create image generation, editing, and variance endpoints compatible with the OpenAI client.

APIs:

POST /images/variations (create_variation)
POST /images/edits (edit)
POST /images/generations (generate)
"""

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from aquilesimage.models import CreateImageRequest, ImagesResponse, CreateImageEditRequest, CreateImageVariationRequest
from aquilesimage.utils import Utils, setup_colored_logger
import asyncio
import logging
from contextlib import asynccontextmanager

logger = setup_colored_logger("Aquiles-Image", logging.INFO)

logger.info("Loading the model...")

model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.total_requests = 0
    app.state.active_inferences = 0
    app.state.metrics_lock = asyncio.Lock()
    app.state.metrics_task = None

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

        logger.info("Lifespan shutdown complete")

app = FastAPI(title="Aquiles-Image", lifespan=lifespan)

@app.middleware("http")
async def count_requests_middleware(request: Request, call_next):
    async with app.state.metrics_lock:
        app.state.total_requests += 1
    response = await call_next(request)
    return response

@app.post("/images/generations", response_model=ImagesResponse, tags=["Generation"])
async def create_image(input_r: CreateImageRequest):
    pass

@app.post("/images/edits", response_model=ImagesResponse, tags=["Edit"])  
async def create_image_edit(input_r: CreateImageEditRequest, image: UploadFile = File(...), mask:  UploadFile | None = File(default=None)):
    pass

@app.post("/images/variations", response_model=ImagesResponse, tags=["Variations"])
async def create_image_variation(input_r: CreateImageVariationRequest, image: UploadFile = File(...)):
    pass


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