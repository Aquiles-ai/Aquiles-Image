import torch
import torch.multiprocessing as mp
import queue
import logging
from typing import Dict, Any
from aquilesimage.utils import setup_colored_logger
import time

logger = setup_colored_logger("Aquiles-Worker", logging.INFO)

def inference_worker_loop(
    gpu_id: int,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    config: Dict[str, Any]
):
    try:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        logger.info(f"[Worker-{gpu_id}] Starting on {device}")
        logger.info(f"[Worker-{gpu_id}] Model: {model_name}")

        pipeline = _load_pipeline_in_worker(model_name, gpu_id, config)
        
        logger.info(f"[Worker-{gpu_id}] Pipeline loaded successfully")

        result_queue.put({
            'type': 'worker_ready',
            'gpu_id': gpu_id,
            'device': device
        })

        while True:
            try:
                batch_data = work_queue.get(timeout=0.1)
                
                if batch_data is None:
                    logger.info(f"[Worker-{gpu_id}] Shutdown signal received")
                    break

                request_id = batch_data.get('request_id')
                prompts = batch_data.get('prompts')
                images = batch_data.get('images', None)
                params = batch_data.get('params', {})
                
                logger.info(
                    f"[Worker-{gpu_id}] Processing batch: "
                    f"request_id={request_id}, "
                    f"prompts={len(prompts)}, "
                    f"has_images={images is not None}"
                )
                
                start_time = time.time()

                try:
                    if images is not None:
                        output = pipeline(
                            prompt=prompts,
                            image=images,
                            **params
                        )
                    else:
                        output = pipeline(
                            prompt=prompts,
                            **params
                        )
                    
                    inference_time = time.time() - start_time
                    
                    logger.info(
                        f"[Worker-{gpu_id}] Batch completed: "
                        f"request_id={request_id}, "
                        f"time={inference_time:.2f}s, "
                        f"images={len(output.images)}"
                    )

                    result_queue.put({
                        'type': 'inference_result',
                        'request_id': request_id,
                        'images': output.images,
                        'success': True,
                        'inference_time': inference_time,
                        'gpu_id': gpu_id
                    })
                    
                except Exception as inference_error:
                    logger.error(
                        f"[Worker-{gpu_id}] Inference error: {inference_error}",
                        exc_info=True
                    )
                    
                    result_queue.put({
                        'type': 'inference_result',
                        'request_id': request_id,
                        'error': str(inference_error),
                        'success': False,
                        'gpu_id': gpu_id
                    })
                
            except queue.Empty:
                continue
                
            except Exception as e:
                logger.error(
                    f"[Worker-{gpu_id}] Unexpected error in main loop: {e}",
                    exc_info=True
                )
                continue
        
        logger.info(f"[Worker-{gpu_id}] Shutting down gracefully")
        
    except Exception as e:
        logger.error(
            f"[Worker-{gpu_id}] FATAL ERROR during initialization: {e}",
            exc_info=True
        )
        result_queue.put({
            'type': 'worker_error',
            'gpu_id': gpu_id,
            'error': str(e)
        })


def _load_pipeline_in_worker(model_name: str, gpu_id: int, config: Dict[str, Any]):
    from aquilesimage.pipelines import ModelPipelineInit
    from aquilesimage.models import ImageModel

    torch.set_float32_matmul_precision("high")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True

    low_vram = config.get('low_vram', False)
    auto_pipeline = config.get('auto_pipeline', False)
    device_map_flux2 = config.get('device_map')
    
    logger.info(f"[Worker-{gpu_id}] Initializing pipeline...")

    if auto_pipeline:
        initializer = ModelPipelineInit(
            model=model_name,
            auto_pipeline=True,
            dist_inf=False
        )
    elif device_map_flux2 == 'cuda' and model_name == ImageModel.FLUX_2_4BNB:
        initializer = ModelPipelineInit(
            model=model_name,
            device_map_flux2='cuda',
            dist_inf=False
        )
    else:
        initializer = ModelPipelineInit(
            model=model_name,
            low_vram=low_vram,
            dist_inf=False
        )

    model_pipeline = initializer.initialize_pipeline()
    model_pipeline.start()
    
    logger.info(f"[Worker-{gpu_id}] Pipeline started successfully")

    return model_pipeline.pipeline