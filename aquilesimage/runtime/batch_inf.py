import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
from aquilesimage.utils import setup_colored_logger
import logging
from aquilesimage.runtime.distributed_inference import DistributedCoordinator

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
        )

class BatchPipeline:    
    def __init__(
        self,
        request_scoped_pipeline: Any,
        max_batch_size: int = 4,
        batch_timeout: float = 0.5,
        worker_sleep: float = 0.05,
        is_dist: bool = False,
        device_ids: Optional[List[str]] = None,
        max_batch_sizes: Optional[List[int]] = None,
    ):
        self.pipeline = request_scoped_pipeline
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.worker_sleep = worker_sleep
        self.is_dist = is_dist
        
        if self.is_dist:
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
        
        logger.info(f"BatchCoordinator configuration:")
        logger.info(f"  max_batch_size={max_batch_size}")
        logger.info(f"  batch_timeout={batch_timeout}s")
        logger.info(f"  worker_sleep={worker_sleep}s")

    async def start(self):
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._batch_worker_loop())
            logger.info("Batch worker started")

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
                await asyncio.sleep(self.worker_sleep)
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
        
        if self.is_dist:
            try:
                device_stats = await self.dist_cor.wait_for_available_device(timeout=30.0)
                device_to_use = device_stats.id

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
                images = images[0]
        
        params = group[0].params.copy()
        params['device'] = device_to_use

        total_expected_images = sum(req.num_images for req in group)
        logger.info(f"Group expects {total_expected_images} total images")
        
        try:
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
                    f"Expected {total_expected_images} images, got {len(output.images)}. "
                    f"Output mapping is BROKEN!"
                )

            image_idx = 0
            for i, req in enumerate(group):
                req_images = output.images[image_idx:image_idx + req.num_images]
                logger.info(f"  [{i}] {req.id} → images[{image_idx}:{image_idx + req.num_images}]")

                if req.num_images == 1:
                    req.future.set_result(req_images[0])
                else:
                    req.future.set_result(req_images)
            
                image_idx += req.num_images

            if self.is_dist and device_stats:
                device_stats.complete_batch(total_expected_images)
                logger.info(f"Device {device_to_use} stats updated - completed {total_expected_images} images")
            
            self.total_batches += 1
            self.total_images += total_expected_images
            self.total_complete += 1
            
            logger.info(
                f"Group completed on {device_to_use}: {total_expected_images} images "
                f"(total_batches={self.total_batches}, "
                f"total_images={self.total_images})"
            )
        
        except Exception as e:
            logger.error(f"X Batch inference failed on {device_to_use}: {e}")
            logger.error(f"  Params: {params}")
            logger.error(f"  Group size: {len(group)}")

            if self.is_dist and device_stats:
                device_stats.register_error(str(e))
                logger.info(f"Device {device_to_use} error registered")
            
            self.total_failed += 1 

            for i, req in enumerate(group):
                logger.error(f"  [{i}] {req.id} → FAILED")
                if not req.future.done():
                    req.future.set_exception(e)

    async def stop(self):
        self.shutdown = True
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