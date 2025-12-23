from aquilesimage.models import VideoResource, VideoStatus, CreateVideoBody, VideoQuality
import asyncio
import time
from typing import Dict, Optional, Any
from collections import deque
from dataclasses import dataclass, field
import uuid
from aquilesimage.utils.utils_video import get_path_save_video
from fastapi.concurrency import run_in_threadpool
import random

@dataclass
class VideoTask:
    id: str
    model: str
    prompt: str
    size: Optional[str]
    seconds: Optional[str]
    quality: VideoQuality
    status: VideoStatus = VideoStatus.queued
    progress: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    error: Optional[dict] = None
    video_path: Optional[str] = None
    
    def to_video_resource(self) -> VideoResource:
        return VideoResource(
            id=self.id,
            object="video",
            model=self.model,
            status=self.status,
            progress=self.progress,
            created_at=self.created_at,
            size=self.size,
            seconds=self.seconds,
            quality=self.quality,
            error=self.error,
            prompt=self.prompt
        )

class VideoTaskGeneration:
    def __init__(
        self,
        pipeline: Any,
        max_concurrent_tasks: int = 1,
        enable_queue: bool = False
    ):
        self.pipeline = pipeline
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_queue = enable_queue

        self.tasks: Dict[str, VideoTask] = {}

        self.queue: deque[str] = deque()

        self.active_tasks: set[str] = set()

        self.lock = asyncio.Lock()

        self.worker_task: Optional[asyncio.Task] = None

    async def start(self):
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        while True:
            try:
                async with self.lock:
                    if (len(self.active_tasks) < self.max_concurrent_tasks 
                        and len(self.queue) > 0):
                        task_id = self.queue.popleft()
                        self.active_tasks.add(task_id)

                        asyncio.create_task(self._process_task(task_id))
                await asyncio.sleep(1)       
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"X Error en el worker de cola: {e}")
                await asyncio.sleep(1)

    async def _process_task(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task:
            return
        try:
            task.status = VideoStatus.processing
            task.progress = 0
            
            # Simulate progress
            for progress in range(0, 101, 10):
                task.progress = progress
                await asyncio.sleep(0.5)
            
            video_output = await self._generate_video(task)
            
            task.status = VideoStatus.completed
            task.progress = 100
            
        except Exception as e:
            task.status = VideoStatus.failed
            task.error = {
                "code": "generation_error",
                "message": str(e)
            }
            print(f"X Error processing task {task_id}: {e}")
        finally:
            async with self.lock:
                self.active_tasks.discard(task_id)

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None

    async def get_task(self, task_id: str) -> Optional[VideoResource]:
        task = self.tasks.get(task_id)
        return task.to_video_resource() if task else None

    async def create_task(self, request: CreateVideoBody) -> VideoResource:
        async with self.lock:
            task_id = f"video_{uuid.uuid4().hex[:12]}"

            task = VideoTask(
                id=task_id,
                model=request.model,
                prompt=request.prompt,
                size=request.size,
                seconds=request.seconds,
                quality=request.quality or VideoQuality.standard,
                video_path=get_path_save_video(task_id)
            )
            
            self.tasks[task_id] = task

            if len(self.active_tasks) < self.max_concurrent_tasks:
                self.active_tasks.add(task_id)
                asyncio.create_task(self._process_task(task_id))
            else:
                if not self.enable_queue:
                    del self.tasks[task_id]
                    raise Exception(
                        f"The limit of {self.max_concurrent_tasks} concurrent tasks has been reached. "
                        "Please try again later."
                    )

                self.queue.append(task_id)
            
            return task.to_video_resource()

    async def list_tasks(
        self, limit: int = 20,
        after: Optional[str] = None) -> tuple[list[VideoResource], bool]:
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        if after:
            start_idx = 0
            for idx, task in enumerate(sorted_tasks):
                if task.id == after:
                    start_idx = idx + 1
                    break
            sorted_tasks = sorted_tasks[start_idx:]
        limited_tasks = sorted_tasks[:limit]
        has_more = len(sorted_tasks) > limit
        return [task.to_video_resource() for task in limited_tasks], has_more

    async def delete_task(self, task_id: str) -> bool:
        async with self.lock:
            if task_id in self.tasks:
                if task_id in self.queue:
                    self.queue.remove(task_id)
                del self.tasks[task_id]
                return True
            return False

    async def _generate_video(self, task: VideoTask) -> Any:
        await run_in_threadpool(self.pipeline.generate,
            seed=random.randint(1, 1000),
            prompt=task.prompt,
            save_result_path=task.video_path,
            negative_prompt="No deformities",
        )

    def get_stats(self) -> dict:
        return {
            "total_tasks": len(self.tasks),
            "queued": len(self.queue),
            "processing": len(self.active_tasks),
            "completed": sum(1 for t in self.tasks.values() if t.status == VideoStatus.completed),
            "failed": sum(1 for t in self.tasks.values() if t.status == VideoStatus.failed),
            "max_concurrent": self.max_concurrent_tasks
        }

    async def get_path_video(self, task_id):
        task = self.tasks.get(task_id)
        return task.video_path