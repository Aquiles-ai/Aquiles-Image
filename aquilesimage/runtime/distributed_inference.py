import torch
from dataclasses import dataclass
import asyncio
from typing import Optional, List
import time
from aquilesimage.utils import setup_colored_logger
import logging

logger = setup_colored_logger("Aquiles-Image-DistributedCoordinator", logging.INFO)

def get_device_count():
    return torch.cuda.device_count()

@dataclass
class DistStats:
    id: str

    is_available: bool = True
    is_processing: bool = False
    
    batch_size: int = 1            
    max_batch_size: int = 4          

    images_processing: int = 0    
    images_completed: int = 0    
    total_batches_processed: int = 0

    avg_batch_time: float = 0.0  
    last_batch_start: float = 0.0 

    error_count: int = 0
    last_error: Optional[str] = None
    
    @property
    def can_accept_batch(self) -> bool:
        return self.is_available and not self.is_processing
    
    @property
    def estimated_load(self) -> float:
        if not self.is_processing:
            return 0.0
        elapsed = time.time() - self.last_batch_start
        if self.avg_batch_time > 0:
            return elapsed / self.avg_batch_time
        return 0.5 
    
    def start_batch(self, num_images: int):
        self.is_processing = True
        self.images_processing = num_images
        self.last_batch_start = time.time()
    
    def complete_batch(self, num_images: int):
        batch_time = time.time() - self.last_batch_start
        
        if self.avg_batch_time == 0.0:
            self.avg_batch_time = batch_time
        else:
            self.avg_batch_time = (self.avg_batch_time * 0.7) + (batch_time * 0.3)
        
        self.images_completed += num_images
        self.total_batches_processed += 1
        self.images_processing = 0
        self.is_processing = False
    
    def register_error(self, error_msg: str):
        self.error_count += 1
        self.last_error = error_msg
        self.is_processing = False
        self.images_processing = 0

class DistributedCoordinator:    
    def __init__(self, device_ids: List[str], max_batch_sizes: Optional[List[int]] = None):
        if max_batch_sizes is None:
            max_batch_sizes = [4] * len(device_ids)
        
        self.devices = [
            DistStats(id=dev_id, max_batch_size=max_batch)
            for dev_id, max_batch in zip(device_ids, max_batch_sizes)
        ]
        
        self.lock = asyncio.Lock()
        self.device_available_event = asyncio.Event()
        self.device_available_event.set()
        
        logger.info(f"DistributedCoordinator initialized with {len(self.devices)} devices:")
        for dev in self.devices:
            logger.info(f"  {dev.id}: max_batch_size={dev.max_batch_size}")
    
    async def get_best_device(self) -> Optional[DistStats]:
        async with self.lock:
            available = [d for d in self.devices if d.can_accept_batch]
            
            if not available:
                self.device_available_event.clear()
                return None

            best = min(available, key=lambda d: (
                d.is_processing,
                d.estimated_load,
                d.images_completed
            ))
            
            return best
    
    async def wait_for_available_device(self, timeout: float = 30.0) -> DistStats:
        start_time = time.time()

        device = await self.get_best_device()
        if device:
            return device

        while time.time() - start_time < timeout:
            remaining = timeout - (time.time() - start_time)
            
            try:
                await asyncio.wait_for(
                    self.device_available_event.wait(),
                    timeout=min(remaining, 0.05)
                )
            except asyncio.TimeoutError:
                pass
            
            device = await self.get_best_device()
            if device:
                return device
        
        raise TimeoutError("No devices are available after waiting")
    
    def notify_device_available(self):
        self.device_available_event.set()
    
    def get_stats_summary(self) -> dict:
        stats = {}
        for dev in self.devices:
            stats[dev.id] = {
                "id": dev.id,
                "available": dev.is_available,
                "processing": dev.is_processing,
                "can_accept_batch": dev.can_accept_batch,
                "batch_size": dev.batch_size,
                "max_batch_size": dev.max_batch_size,
                "images_processing": dev.images_processing,
                "images_completed": dev.images_completed,
                "total_batches_processed": dev.total_batches_processed,
                "avg_batch_time": round(dev.avg_batch_time, 2),
                "estimated_load": round(dev.estimated_load, 2),
                "error_count": dev.error_count,
                "last_error": dev.last_error
            }
    
        return stats

    def get_stats_text(self) -> str:
        # For logging
        stats = self.get_stats_summary()
        lines = ["\nDistributed Inference Status:"]
    
        for device_id, info in stats.items():
            status = "Processing" if info["processing"] else "Available"
            lines.append(
                f"  {device_id}: {status} | "
                f"Processing: {info['images_processing']} imgs | "
                f"Completed: {info['images_completed']} | "
                f"Batches: {info['total_batches_processed']} | "
                f"Avg: {info['avg_batch_time']:.2f}s"
            )
    
        return "\n".join(lines)