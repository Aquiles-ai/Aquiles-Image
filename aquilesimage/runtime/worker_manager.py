import torch
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional
import logging
from aquilesimage.utils import setup_colored_logger
import time

logger = setup_colored_logger("Aquiles-WorkerManager", logging.INFO)


class WorkerManager:
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        num_workers: Optional[int] = None
    ):
        self.model_name = model_name
        self.config = config

        if num_workers is None:
            self.num_workers = torch.cuda.device_count()
        else:
            self.num_workers = min(num_workers, torch.cuda.device_count())
        
        if self.num_workers == 0:
            raise RuntimeError("No CUDA devices available for workers")
        
        logger.info(f"WorkerManager: {self.num_workers} workers will be spawned")

        self.work_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []
        self.worker_processes: List[mp.Process] = []
        self.device_ids: List[str] = []
        
        self._started = False
    
    def start(self):
        if self._started:
            logger.warning("WorkerManager already started")
            return
        
        logger.info("Starting workers...")
        
        from aquilesimage.runtime.worker_process import inference_worker_loop
        
        for gpu_id in range(self.num_workers):
            device = f"cuda:{gpu_id}"
            self.device_ids.append(device)

            work_q = mp.Queue(maxsize=20)
            result_q = mp.Queue()
            
            self.work_queues.append(work_q)
            self.result_queues.append(result_q)
            
            # Spawn worker process
            process = mp.Process(
                target=inference_worker_loop,
                args=(gpu_id, work_q, result_q, self.model_name, self.config),
                daemon=False,
                name=f"InferenceWorker-{gpu_id}"
            )
            
            process.start()
            self.worker_processes.append(process)
            
            logger.info(f"Worker process started: PID={process.pid}, GPU={device}")

        self._wait_for_workers_ready()
        
        self._started = True
        logger.info("All workers ready!")
    
    def _wait_for_workers_ready(self, timeout: float = 300.0):
        logger.info("Waiting for workers to initialize...")
        
        ready_count = 0
        start_time = time.time()
        
        while ready_count < self.num_workers:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Workers did not initialize within {timeout}s. "
                    f"Ready: {ready_count}/{self.num_workers}"
                )

            for idx, result_q in enumerate(self.result_queues):
                try:
                    msg = result_q.get(timeout=0.1)
                    
                    if msg.get('type') == 'worker_ready':
                        gpu_id = msg.get('gpu_id')
                        logger.info(f"Worker {gpu_id} is ready")
                        ready_count += 1
                    
                    elif msg.get('type') == 'worker_error':
                        gpu_id = msg.get('gpu_id')
                        error = msg.get('error')
                        raise RuntimeError(
                            f"Worker {gpu_id} failed to initialize: {error}"
                        )
                
                except Exception:
                    continue
        
        logger.info(f"All {self.num_workers} workers initialized successfully")
    
    def stop(self):
        if not self._started:
            return
        
        logger.info("Stopping workers...")

        for idx, work_q in enumerate(self.work_queues):
            try:
                work_q.put(None, timeout=1.0)
                logger.info(f"Shutdown signal sent to worker {idx}")
            except Exception as e:
                logger.warning(f"Failed to send shutdown to worker {idx}: {e}")

        for idx, process in enumerate(self.worker_processes):
            try:
                process.join(timeout=10.0)
                if process.is_alive():
                    logger.warning(f"Worker {idx} did not stop, terminating...")
                    process.terminate()
                    process.join(timeout=5.0)
                    if process.is_alive():
                        logger.error(f"Worker {idx} still alive, killing...")
                        process.kill()
                else:
                    logger.info(f"Worker {idx} stopped gracefully")
            except Exception as e:
                logger.error(f"Error stopping worker {idx}: {e}")
        
        self._started = False
        logger.info("All workers stopped")
    
    def get_queues(self):
        return self.work_queues, self.result_queues
    
    def get_device_ids(self):
        return self.device_ids