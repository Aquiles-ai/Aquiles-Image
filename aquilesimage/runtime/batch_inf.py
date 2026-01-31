import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
from aquilesimage.utils import setup_colored_logger
import logging
from aquilesimage.runtime.distributed_inference import DistributedCoordinator
import torch.multiprocessing as mp
import queue

logger = setup_colored_logger("Aquiles-Image-BatchPipeline", logging.INFO)

@dataclass
class PendingRequest:
    id: str
    prompt: str
    image: Optional[Any] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    num_images: int = 1 
    
    def params_key(self) -> Tuple:
        has_image = self.image is not None
        num_input_images = 0
        if self.image is not None:
            if isinstance(self.image, list):
                num_input_images = len(self.image)
            else:
                num_input_images = 1
        return (
            self.params.get('height', 1024),
            self.params.get('width', 1024),
            self.params.get('num_inference_steps', 30),
            self.params.get('device', 'cuda'),
            has_image,
            num_input_images,
            self.num_images,
            self.params.get('use_glm', False)
        )

class BatchPipeline:    
    def __init__(
        self,
        request_scoped_pipeline: Any = None,
        work_queues: Optional[List[mp.Queue]] = None,
        result_queues: Optional[List[mp.Queue]] = None,
        max_batch_size: int = 4,
        batch_timeout: float = 0.5,
        worker_sleep: float = 0.001,
        is_dist: bool = False,
        device_ids: Optional[List[str]] = None,
        max_batch_sizes: Optional[List[int]] = None,
    ):
        self.pipeline = request_scoped_pipeline
        self.work_queues = work_queues
        self.result_queues = result_queues
        
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.worker_sleep = worker_sleep
        self.is_dist = is_dist
        
        if self.is_dist:
            if work_queues is None or result_queues is None:
                raise ValueError("work_queues and result_queues required for distributed mode")
            if device_ids is None or len(device_ids) == 0:
                raise ValueError("device_ids required for distributed mode")
            
            self.dist_cor = DistributedCoordinator(
                device_ids=device_ids,
                max_batch_sizes=max_batch_sizes
            )
            logger.info(f"BatchPipeline initialized in DISTRIBUTED mode with {len(device_ids)} devices")
        else:
            self.dist_cor = None
            logger.info("BatchPipeline initialized in SINGLE-DEVICE mode")

        self.pending: deque[PendingRequest] = deque()
        self.lock = asyncio.Lock()
        self.new_request_event = asyncio.Event()

        self.processing = False
        self.active_batches = 0
        self.active_batches_lock = asyncio.Lock()
        
        self.worker_task: Optional[asyncio.Task] = None
        self.shutdown = False

        self.total_requests = 0
        self.total_batches = 0
        self.total_images = 0
        self.total_complete = 0
        self.total_failed = 0

        self.pending_results: Dict[str, asyncio.Future] = {}
        self.result_lock = asyncio.Lock()
        self.result_listener_task: Optional[asyncio.Task] = None
        
        logger.info(f"BatchCoordinator configuration:")
        logger.info(f"  max_batch_size={max_batch_size}")
        logger.info(f"  batch_timeout={batch_timeout}s")
        logger.info(f"  worker_sleep={worker_sleep}s")

    async def start(self):
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._batch_worker_loop())
            logger.info("Batch worker started")

        if self.is_dist and self.result_listener_task is None:
            self.result_listener_task = asyncio.create_task(self._result_listener_loop())
            logger.info("Result listener started")

    async def _result_listener_loop(self):
        logger.info("Result listener loop started")
        
        while not self.shutdown:
            try:
                for idx, result_q in enumerate(self.result_queues):
                    try:
                        result_data = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda q=result_q: q.get(block=False)
                        )
                        
                        if result_data.get('type') == 'inference_result':
                            request_id = result_data.get('request_id')
                            
                            async with self.result_lock:
                                if request_id in self.pending_results:
                                    future = self.pending_results[request_id]
                                    
                                    if result_data.get('success'):
                                        future.set_result(result_data)
                                    else:
                                        error_msg = result_data.get('error', 'Unknown error')
                                        future.set_exception(RuntimeError(error_msg))
                                    
                                    del self.pending_results[request_id]
                                    
                                    gpu_id = result_data.get('gpu_id')
                                    logger.info(f"Result received from worker {gpu_id}: request_id={request_id}")
                                else:
                                    logger.warning(f"Received result for unknown request_id: {request_id}")
                    
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing result from queue {idx}: {e}")
                        continue

                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                logger.info("Result listener cancelled")
                break
            except Exception as e:
                logger.error(f"Error in result listener: {e}")
                await asyncio.sleep(0.1)

    async def submit(
        self,
        prompt: str,
        image: Optional[Any] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        device: Optional[str] = None,
        request_id: Optional[str] = None,
        timeout: float = 60.0,
        num_images_per_prompt: int = 1,
        **kwargs
    ) -> Any:
        req_id = request_id or str(uuid.uuid4())[:8]

        logger.info(f"submit (BatchPipeline) - req_id:{req_id} num_images_per_prompt:{num_images_per_prompt}")
        
        request = PendingRequest(
            id=req_id,
            prompt=prompt,
            image=image,
            params={
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps,
                'device': device or 'cuda',
                'num_images_per_prompt': num_images_per_prompt,
                **kwargs
            },
            timestamp=time.time(),
            num_images=num_images_per_prompt
        )
        
        async with self.lock:
            self.pending.append(request)
            queue_size = len(self.pending)
            self.total_requests += 1
        self.new_request_event.set()
        
        request_type = "I2I" if image is not None else "T2I"
        logger.info(
            f"Request {req_id} queued ({request_type}, "
            f"queue_size={queue_size}, prompt='{prompt[:50]}...')"
        )

        try:
            result = await asyncio.wait_for(request.future, timeout=timeout)
            logger.info(f"Request {req_id} completed")
            return result
        except asyncio.TimeoutError:
            logger.error(f"X Request {req_id} timed out after {timeout}s")
            raise

    async def _batch_worker_loop(self):
        logger.info("Batch worker loop started")
        
        while not self.shutdown:
            try:
                try:
                    await asyncio.wait_for(
                        self.new_request_event.wait(), 
                        timeout=self.worker_sleep
                    )
                except asyncio.TimeoutError:
                    pass
                
                self.new_request_event.clear()

                if not self.is_dist and self.processing:
                    continue

                async with self.lock:
                    if len(self.pending) == 0:
                        continue

                    oldest = self.pending[0]
                    age = time.time() - oldest.timestamp

                    should_process = (
                        len(self.pending) >= self.max_batch_size or
                        age >= self.batch_timeout
                    )
                    
                    if not should_process:
                        continue

                    batch = []
                    extracted = 0
                    max_extract = min(self.max_batch_size, len(self.pending))
                    
                    while extracted < max_extract:
                        batch.append(self.pending.popleft())
                        extracted += 1
                    
                    if not self.is_dist:
                        self.processing = True

                if self.is_dist:
                    asyncio.create_task(self._process_batch_async(batch))
                else:
                    try:
                        await self._process_batch(batch)
                    finally:
                        self.processing = False
                    
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"X Error in worker loop: {e}")
                if not self.is_dist:
                    self.processing = False

    async def _process_batch_async(self, batch: List[PendingRequest]):
        async with self.active_batches_lock:
            self.active_batches += 1
            logger.info(f"[DIST] Starting batch processing (active_batches={self.active_batches})")
        
        try:
            await self._process_batch(batch)
        except Exception as e:
            logger.error(f"X Error in async batch processing: {e}")
        finally:
            async with self.active_batches_lock:
                self.active_batches -= 1
                logger.info(f"[DIST] Finished batch processing (active_batches={self.active_batches})")

    async def _process_batch(self, batch: List[PendingRequest]):
        if not batch:
            return
        
        logger.info(f"Processing batch of {len(batch)} requests")

        groups = self._group_by_params(batch)
        
        logger.info(f"  Grouped into {len(groups)} compatible batches")

        if len(groups) > 1:
            tasks = [
                asyncio.create_task(self._process_group(group, params_key))
                for params_key, group in groups.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            for params_key, group in groups.items():
                await self._process_group(group, params_key)

    def _group_by_params(
        self, 
        batch: List[PendingRequest]
    ) -> Dict[Tuple, List[PendingRequest]]:
        groups: Dict[Tuple, List[PendingRequest]] = {}
        
        for req in batch:
            key = req.params_key()
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(req)
        
        return groups

    async def _process_group(
        self, 
        group: List[PendingRequest],
        params_key: Tuple
    ):
        if not group:
            return

        device_to_use = None
        device_stats = None
        worker_idx = None
        
        if self.is_dist:
            try:
                device_stats = await self.dist_cor.wait_for_available_device(timeout=30.0)
                device_to_use = device_stats.id
                worker_idx = int(device_to_use.split(':')[1])  # cuda:0 -> 0

                total_images = sum(req.num_images for req in group)
                device_stats.start_batch(total_images)
                
                logger.info(f"Assigned device {device_to_use} for batch of {len(group)} requests ({total_images} images)")
                
            except TimeoutError as e:
                logger.error(f"X No devices available after timeout")
                for req in group:
                    if not req.future.done():
                        req.future.set_exception(e)
                self.total_failed += 1
                return
        else:
            device_to_use = group[0].params.get('device', 'cuda')

        has_images = group[0].image is not None
        batch_type = "Image-to-Image" if has_images else "Text-to-Image"

        logger.info(f"Processing {batch_type} group on device {device_to_use} with params {params_key}:")
        for i, req in enumerate(group):
            logger.info(f"  [{i}] {req.id}: '{req.prompt[:50]}...'")

        prompts = [r.prompt for r in group]

        images = None
        if has_images:
            images = []
            for r in group:
                if isinstance(r.image, list):
                    images.extend(r.image)
                else:
                    images.append(r.image)

            if len(images) == 1:

                if group[0].params.get('use_glm') is True:
                    images = [images[0]]
                    group[0].params.pop('use_glm', None)
                else:
                    images = images[0]
        
        group[0].params.pop('use_glm', None)

        params = group[0].params.copy()
        
        total_expected_images = sum(req.num_images for req in group)
        logger.info(f"Group expects {total_expected_images} total images")
        
        try:
            if self.is_dist:
                
                request_id = str(uuid.uuid4())

                batch_data = {
                    'request_id': request_id,
                    'prompts': prompts,
                    'images': images,
                    'params': {
                        'height': params['height'],
                        'width': params['width'],
                        'num_inference_steps': params['num_inference_steps'],
                        'num_images_per_prompt': params['num_images_per_prompt'],
                        **{k: v for k, v in params.items() 
                           if k not in ['height', 'width', 'num_inference_steps', 'device', 'num_images_per_prompt', 'use_glm']}
                    }
                }

                result_future = asyncio.Future()
                async with self.result_lock:
                    self.pending_results[request_id] = result_future

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.work_queues[worker_idx].put,
                    batch_data
                )
                
                logger.info(f"Batch sent to worker {worker_idx}: request_id={request_id}")

                try:
                    result_data = await asyncio.wait_for(result_future, timeout=600.0)
                except asyncio.TimeoutError:
                    logger.error(f"X Worker {worker_idx} timed out for request_id={request_id}")
                    async with self.result_lock:
                        self.pending_results.pop(request_id, None)
                    raise
                
                output_images = result_data['images']
                
                if len(output_images) != total_expected_images:
                    raise RuntimeError(
                        f"X CRITICAL: Batch size mismatch! "
                        f"Expected {total_expected_images} images, got {len(output_images)}"
                    )
                
                image_idx = 0
                for i, req in enumerate(group):
                    req_images = output_images[image_idx:image_idx + req.num_images]
                    
                    if req.num_images == 1:
                        req.future.set_result(req_images[0])
                    else:
                        req.future.set_result(req_images)
                
                    image_idx += req.num_images
                
                if device_stats:
                    device_stats.complete_batch(total_expected_images)
                    self.dist_cor.notify_device_available()
                
                self.total_batches += 1
                self.total_images += total_expected_images
                self.total_complete += 1
                
                logger.info(
                    f"Group completed on {device_to_use}: {total_expected_images} images "
                    f"(total_batches={self.total_batches}, total_images={self.total_images})"
                )
            
            else:                
                from fastapi.concurrency import run_in_threadpool
                
                def batch_infer():
                    if images is not None:
                        return self.pipeline.generate_batch(
                            prompts=prompts,
                            image=images,
                            height=params['height'],
                            width=params['width'],
                            num_inference_steps=params['num_inference_steps'],
                            device=device_to_use,
                            num_images_per_prompt=params['num_images_per_prompt'],
                            **{k: v for k, v in params.items() 
                                if k not in ['height', 'width', 'num_inference_steps', 'device', 'image', 'images', 'num_images_per_prompt']}
                        )
                    else:
                        return self.pipeline.generate_batch(
                            prompts=prompts,
                            height=params['height'],
                            width=params['width'],
                            num_inference_steps=params['num_inference_steps'],
                            device=device_to_use,
                            num_images_per_prompt=params['num_images_per_prompt'],
                            **{k: v for k, v in params.items() 
                                if k not in ['height', 'width', 'num_inference_steps', 'device', 'image', 'images', 'num_images_per_prompt']}
                        )

                output = await run_in_threadpool(batch_infer)

                if len(output.images) != total_expected_images:
                    raise RuntimeError(
                        f"X CRITICAL: Batch size mismatch! "
                        f"Expected {total_expected_images} images, got {len(output.images)}"
                    )

                image_idx = 0
                for i, req in enumerate(group):
                    req_images = output.images[image_idx:image_idx + req.num_images]
                    
                    if req.num_images == 1:
                        req.future.set_result(req_images[0])
                    else:
                        req.future.set_result(req_images)
                
                    image_idx += req.num_images

                if self.is_dist and device_stats:
                    device_stats.complete_batch(total_expected_images)
                    self.dist_cor.notify_device_available()
                
                self.total_batches += 1
                self.total_images += total_expected_images
                self.total_complete += 1
                
                logger.info(
                    f"Group completed on {device_to_use}: {total_expected_images} images "
                    f"(total_batches={self.total_batches}, total_images={self.total_images})"
                )
        
        except Exception as e:
            logger.error(f"X Batch inference failed on {device_to_use}: {e}", exc_info=True)

            if self.is_dist and device_stats:
                device_stats.register_error(str(e))
                self.dist_cor.notify_device_available()
            
            self.total_failed += 1 

            for i, req in enumerate(group):
                if not req.future.done():
                    req.future.set_exception(e)

    async def stop(self):
        self.shutdown = True
        
        if self.result_listener_task:
            self.result_listener_task.cancel()
            try:
                await self.result_listener_task
            except asyncio.CancelledError:
                pass
            logger.info("Result listener stopped")
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Batch worker stopped")

    async def get_stats(self) -> dict:
        if self.is_dist:
            dist_stats = self.dist_cor.get_stats_summary()

            async with self.lock:
                queued = len(self.pending)
            
            async with self.active_batches_lock:
                active = self.active_batches
            
            return {
                "mode": "distributed",
                "devices": dist_stats,
                "global": {
                    "total_requests": self.total_requests,
                    "total_batches": self.total_batches,
                    "total_images": self.total_images,
                    "queued": queued,
                    "active_batches": active,
                    "completed": self.total_complete,
                    "failed": self.total_failed,
                    "processing": self.processing
                }
            }
        else:
            async with self.lock:
                queued = len(self.pending)
                total_requests = self.total_requests
                processing = self.processing

            return {
                "mode": "single-device",
                "total_requests": total_requests,
                "total_batches": self.total_batches,
                "total_images": self.total_images,
                "queued": queued,
                "completed": self.total_complete,
                "failed": self.total_failed,
                "processing": processing,
                "available": not processing
            }
    
    def get_stats_text(self) -> str:
        if self.is_dist:
            return self.dist_cor.get_stats_text()
        else:
            return (
                f"\nBatch Pipeline Status (Single-Device):\n"
                f"  Total Requests: {self.total_requests}\n"
                f"  Total Batches: {self.total_batches}\n"
                f"  Total Images: {self.total_images}\n"
                f"  Queued: {len(self.pending)}\n"
                f"  Completed: {self.total_complete}\n"
                f"  Failed: {self.total_failed}\n"
                f"  Processing: {self.processing}"
            )