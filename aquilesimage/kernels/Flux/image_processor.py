# The base code was taken: https://github.com/huggingface/diffusers/blob/main/src/diffusers/image_processor.py (It has been modified)

import warnings
import threading
import queue
import gc
import psutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union, Callable, Any
import numpy as np
import PIL.Image
import torch
from PIL import Image, ImageFilter, ImageOps

from aquilesimage.kernels.configs import ConfigMixin, register_to_config
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.constants import CONFIG_NAME
from diffusers.utils.pil_utils import PIL_INTERPOLATION


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

PipelineDepthInput = PipelineImageInput


@dataclass
class PostProcessJob:
    tensor: torch.Tensor
    output_type: str
    do_denormalize: Optional[List[bool]]
    callback: Optional[Callable]
    request_id: str
    priority: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()


class AsyncPostProcessor:
    def __init__(self, max_concurrent_jobs: int = 3, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent_jobs
        self.processing_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.active_jobs = 0
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs, thread_name_prefix="PostProcess")
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.worker_thread.start()
        
        # Statistics
        self.jobs_processed = 0
        self.jobs_failed = 0
    
    def _process_worker(self):
        while not self.shutdown_event.is_set():
            try:
                try:
                    priority_tuple = self.processing_queue.get(timeout=1.0)
                    priority, job = priority_tuple
                except queue.Empty:
                    continue
                
                with self.lock:
                    self.active_jobs += 1
                
                try:
                    future = self.executor.submit(self._process_single_job, job)
                    result = future.result(timeout=30)
                    
                    if job.callback and result is not None:
                        job.callback(result, job.request_id)
                    
                    self.jobs_processed += 1
                    
                except Exception as e:
                    print(f"Error processing postprocess job {job.request_id}: {e}")
                    if job.callback:
                        job.callback(None, job.request_id, error=e)
                    self.jobs_failed += 1
                
                finally:
                    with self.lock:
                        self.active_jobs -= 1
                    self.processing_queue.task_done()
                    
            except Exception as e:
                print(f"Post-process worker error: {e}")
    
    def _process_single_job(self, job: PostProcessJob) -> Any:
        try:
            original_device = job.tensor.device
            if job.tensor.device.type == 'cuda':
                cpu_tensor = job.tensor.cpu()
                del job.tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                cpu_tensor = job.tensor
            
            if job.output_type == "latent":
                return cpu_tensor.to(original_device) if job.output_type == "pt" else cpu_tensor
                
            elif job.output_type == "pt":
                if job.do_denormalize is not None:
                    cpu_tensor = self._denormalize_conditionally_static(cpu_tensor, job.do_denormalize)
                else:
                    cpu_tensor = (cpu_tensor * 0.5 + 0.5).clamp(0, 1)
                return cpu_tensor.to(original_device)
                
            elif job.output_type == "np":
                if job.do_denormalize is not None:
                    cpu_tensor = self._denormalize_conditionally_static(cpu_tensor, job.do_denormalize)
                else:
                    cpu_tensor = (cpu_tensor * 0.5 + 0.5).clamp(0, 1)
                return self._pt_to_numpy_optimized(cpu_tensor)
                
            elif job.output_type == "pil":
                if job.do_denormalize is not None:
                    cpu_tensor = self._denormalize_conditionally_static(cpu_tensor, job.do_denormalize)
                else:
                    cpu_tensor = (cpu_tensor * 0.5 + 0.5).clamp(0, 1)
                numpy_images = self._pt_to_numpy_optimized(cpu_tensor)
                return self._numpy_to_pil_optimized(numpy_images)
                
            else:
                raise ValueError(f"Unsupported output_type: {job.output_type}")
                
        except Exception as e:
            print(f"Error in _process_single_job: {e}")
            raise
        finally:
            # Cleanup
            if 'cpu_tensor' in locals():
                del cpu_tensor
            gc.collect()
    
    @staticmethod
    def _denormalize_conditionally_static(images: torch.Tensor, do_denormalize: List[bool]) -> torch.Tensor:
        return torch.stack([
            (images[i] * 0.5 + 0.5).clamp(0, 1) if do_denormalize[i] else images[i] 
            for i in range(images.shape[0])
        ])
    
    @staticmethod
    def _pt_to_numpy_optimized(images: torch.Tensor) -> np.ndarray:
        return images.permute(0, 2, 3, 1).float().numpy()
    
    @staticmethod
    def _numpy_to_pil_optimized(images: np.ndarray) -> List[PIL.Image.Image]:
        if images.ndim == 3:
            images = images[None, ...]
        
        # Convert to uint8 in chunks to save memory
        batch_size = images.shape[0]
        pil_images = []
        
        chunk_size = min(4, batch_size)  # Process max 4 images at once
        for i in range(0, batch_size, chunk_size):
            chunk = images[i:i+chunk_size]
            # Convert chunk to uint8
            chunk_uint8 = (chunk * 255).round().astype("uint8")
            
            # Convert to PIL
            for img in chunk_uint8:
                if img.shape[-1] == 1:
                    pil_images.append(Image.fromarray(img.squeeze(), mode="L"))
                else:
                    pil_images.append(Image.fromarray(img))
            
            # Clean up chunk
            del chunk, chunk_uint8
        
        return pil_images
    
    def schedule_postprocessing(self, tensor: torch.Tensor, output_type: str = "pil", 
                              do_denormalize: Optional[List[bool]] = None,
                              callback: Optional[Callable] = None, request_id: str = "",
                              priority: int = 0) -> bool:
        try:
            job = PostProcessJob(
                tensor=tensor,
                output_type=output_type,
                do_denormalize=do_denormalize,
                callback=callback,
                request_id=request_id,
                priority=priority
            )
            
            priority_tuple = (priority, job.timestamp, job)
            self.processing_queue.put_nowait(priority_tuple)
            return True
            
        except queue.Full:
            print(f"Post-processing queue is full, dropping job {request_id}")
            return False
    
    def get_queue_status(self) -> dict:
        with self.lock:
            return {
                "queue_size": self.processing_queue.qsize(),
                "active_jobs": self.active_jobs,
                "jobs_processed": self.jobs_processed,
                "jobs_failed": self.jobs_failed,
                "max_concurrent": self.max_concurrent
            }
    
    def shutdown(self, timeout: float = 10.0):
        self.shutdown_event.set()
        self.executor.shutdown(wait=True, timeout=timeout)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)


class MemoryAwareBatcher:    
    def __init__(self, target_memory_usage: float = 0.8):
        self.target_memory_usage = target_memory_usage
        self.last_batch_size = 4
        
    def get_optimal_batch_size(self, tensor_shape: Tuple[int, ...], 
                             target_operation: str = "postprocess") -> int:
        try:
            # Get available memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                available_gpu = gpu_memory - gpu_used
                memory_limit = available_gpu * self.target_memory_usage
            else:
                # Use system RAM
                memory_stats = psutil.virtual_memory()
                memory_limit = memory_stats.available * self.target_memory_usage
            
            # Estimate memory per tensor (rough approximation)
            if len(tensor_shape) == 3:  # (C, H, W)
                c, h, w = tensor_shape
                bytes_per_tensor = c * h * w * 4  # float32
                
                # Account for intermediate tensors in processing
                if target_operation == "postprocess":
                    # Need space for: original + denormalized + numpy + pil
                    memory_multiplier = 4
                else:  # preprocess
                    # Need space for: original + resized + converted
                    memory_multiplier = 3
                
                total_memory_per_tensor = bytes_per_tensor * memory_multiplier
                max_batch_size = max(1, int(memory_limit / total_memory_per_tensor))
                
                # Clamp between reasonable bounds
                optimal_batch_size = min(max(1, max_batch_size), 16)
                
                # Use exponential smoothing to avoid oscillations
                self.last_batch_size = int(0.7 * self.last_batch_size + 0.3 * optimal_batch_size)
                return max(1, self.last_batch_size)
                
            return 4 
            
        except Exception as e:
            print(f"Error calculating optimal batch size: {e}")
            return 4 


def is_valid_image(image) -> bool:
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)


def is_valid_image_imagelist(images):
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False


class VaeImageProcessor(ConfigMixin):
    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 4,
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
        # Enhanced parameters
        enable_async_postprocess: bool = True,
        max_concurrent_postprocess: int = 3,
        enable_memory_aware_batching: bool = True,
        enable_threaded_pil_ops: bool = True,
        max_pil_threads: int = 4,
    ):
        super().__init__()
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )
        
        # Enhanced features
        self.enable_async_postprocess = enable_async_postprocess
        self.enable_memory_aware_batching = enable_memory_aware_batching
        self.enable_threaded_pil_ops = enable_threaded_pil_ops
        
        # Initialize async postprocessor
        if self.enable_async_postprocess:
            self.async_postprocessor = AsyncPostProcessor(
                max_concurrent_jobs=max_concurrent_postprocess
            )
        
        # Initialize memory-aware batcher
        if self.enable_memory_aware_batching:
            self.memory_batcher = MemoryAwareBatcher()
        
        # Thread pool for PIL operations
        if self.enable_threaded_pil_ops:
            self.pil_executor = ThreadPoolExecutor(
                max_workers=max_pil_threads, 
                thread_name_prefix="PILOps"
            )
    
    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
         
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @staticmethod
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
         
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)
        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
         
        if images.ndim == 3:
            images = images[..., None]
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
         
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
         
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (images * 0.5 + 0.5).clamp(0, 1)

    
    def _pil_to_numpy_threaded(self, images: List[PIL.Image.Image]) -> np.ndarray:
        if not self.enable_threaded_pil_ops or len(images) < 4:
            return self.pil_to_numpy(images)
        
        def convert_single(img):
            return np.array(img).astype(np.float32) / 255.0
        
        futures = [self.pil_executor.submit(convert_single, img) for img in images]
        numpy_images = [future.result() for future in as_completed(futures)]
        
        return np.stack(numpy_images, axis=0)
    
    def _numpy_to_pil_chunked(self, images: np.ndarray) -> List[PIL.Image.Image]:
        if images.ndim == 3:
            images = images[None, ...]
            
        batch_size = images.shape[0]
        
        if not self.enable_memory_aware_batching or batch_size <= 4:
            return self.numpy_to_pil(images)
        
        # Process in memory-aware chunks
        optimal_chunk_size = self.memory_batcher.get_optimal_batch_size(
            images.shape[1:], "postprocess"
        )
        
        pil_images = []
        for i in range(0, batch_size, optimal_chunk_size):
            chunk = images[i:i+optimal_chunk_size]
            chunk_uint8 = (chunk * 255).round().astype("uint8")
            
            for img in chunk_uint8:
                if img.shape[-1] == 1:
                    pil_images.append(Image.fromarray(img.squeeze(), mode="L"))
                else:
                    pil_images.append(Image.fromarray(img))
            
            del chunk, chunk_uint8 
        
        return pil_images
    
    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        return image.convert("RGB")

    @staticmethod
    def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
        return image.convert("L")

    @staticmethod
    def blur(image: PIL.Image.Image, blur_factor: int = 4) -> PIL.Image.Image:
        return image.filter(ImageFilter.GaussianBlur(blur_factor))

    @staticmethod
    def get_crop_region(mask_image: PIL.Image.Image, width: int, height: int, pad=0):
        mask_image = mask_image.convert("L")
        mask = np.array(mask_image)

        h, w = mask.shape
        crop_left = 0
        for i in range(w):
            if not (mask[:, i] == 0).all():
                break
            crop_left += 1

        crop_right = 0
        for i in reversed(range(w)):
            if not (mask[:, i] == 0).all():
                break
            crop_right += 1

        crop_top = 0
        for i in range(h):
            if not (mask[i] == 0).all():
                break
            crop_top += 1

        crop_bottom = 0
        for i in reversed(range(h)):
            if not (mask[i] == 0).all():
                break
            crop_bottom += 1

        x1, y1, x2, y2 = (
            int(max(crop_left - pad, 0)),
            int(max(crop_top - pad, 0)),
            int(min(w - crop_right + pad, w)),
            int(min(h - crop_bottom + pad, h)),
        )

        ratio_crop_region = (x2 - x1) / (y2 - y1)
        ratio_processing = width / height

        if ratio_crop_region > ratio_processing:
            desired_height = (x2 - x1) / ratio_processing
            desired_height_diff = int(desired_height - (y2 - y1))
            y1 -= desired_height_diff // 2
            y2 += desired_height_diff - desired_height_diff // 2
            if y2 >= mask_image.height:
                diff = y2 - mask_image.height
                y2 -= diff
                y1 -= diff
            if y1 < 0:
                y2 -= y1
                y1 -= y1
            if y2 >= mask_image.height:
                y2 = mask_image.height
        else:
            desired_width = (y2 - y1) * ratio_processing
            desired_width_diff = int(desired_width - (x2 - x1))
            x1 -= desired_width_diff // 2
            x2 += desired_width_diff - desired_width_diff // 2
            if x2 >= mask_image.width:
                diff = x2 - mask_image.width
                x2 -= diff
                x1 -= diff
            if x1 < 0:
                x2 -= x1
                x1 -= x1
            if x2 >= mask_image.width:
                x2 = mask_image.width

        return x1, y1, x2, y2

    def _resize_and_fill(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION["lanczos"])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(
                    resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(
                    resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                    box=(fill_width + src_w, 0),
                )

        return res

    def _resize_and_crop(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION["lanczos"])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    def resize(self, image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], height: int, width: int, resize_mode: str = "default") -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
            
        if isinstance(image, PIL.Image.Image):
            if resize_mode == "default":
                return image.resize(
                    (width, height),
                    resample=PIL_INTERPOLATION[self.config.resample],
                    reducing_gap=self.config.reducing_gap,
                )
            elif resize_mode == "fill":
                return self._resize_and_fill(image, width, height)
            elif resize_mode == "crop":
                return self._resize_and_crop(image, width, height)
            else:
                raise ValueError(f"resize_mode {resize_mode} is not supported")
        elif isinstance(image, torch.Tensor):
            return torch.nn.functional.interpolate(image, size=(height, width))
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(image, size=(height, width))
            return self.pt_to_numpy(image)
        
        return image

    def binarize(self, image: PIL.Image.Image) -> PIL.Image.Image:
        image[image < 0.5] = 0
        image[image >= 0.5] = 1
        return image

    def _denormalize_conditionally(self, images: torch.Tensor, do_denormalize: Optional[List[bool]] = None) -> torch.Tensor:
        if do_denormalize is None:
            return self.denormalize(images) if self.config.do_normalize else images

        if all(do_denormalize):
            return self.denormalize(images)
        elif not any(do_denormalize):
            return images
        else:
            return torch.stack([
                self.denormalize(images[i]) if do_denormalize[i] else images[i] 
                for i in range(images.shape[0])
            ])

    def get_default_height_width(self, image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], height: Optional[int] = None, width: Optional[int] = None) -> Tuple[int, int]:
        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        width, height = (
            x - x % self.config.vae_scale_factor for x in (width, height)
        )
        return height, width

    def preprocess(self, image: PipelineImageInput, height: Optional[int] = None, width: Optional[int] = None, resize_mode: str = "default", crops_coords: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(1)
            else:
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn("Passing `image` as a list of 4d np.ndarray is deprecated.", FutureWarning)
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn("Passing `image` as a list of 4d torch.Tensor is deprecated.", FutureWarning)
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}")
        
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                if self.enable_threaded_pil_ops and len(image) >= 4:
                    futures = [self.pil_executor.submit(lambda img: img.crop(crops_coords), img) for img in image]
                    image = [future.result() for future in futures]
                else:
                    image = [i.crop(crops_coords) for i in image]
            
            if self.config.do_resize:
                height, width = self.get_default_height_width(image[0], height, width)
                if self.enable_threaded_pil_ops and len(image) >= 4:
                    # Parallel resize operations
                    futures = [
                        self.pil_executor.submit(self.resize, img, height, width, resize_mode) 
                        for img in image
                    ]
                    image = [future.result() for future in futures]
                else:
                    image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            
            if self.enable_threaded_pil_ops:
                image = self._pil_to_numpy_threaded(image)
            else:
                image = self.pil_to_numpy(image)
            image = self.numpy_to_pt(image)

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = self.numpy_to_pt(image)
            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)
            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)
            
            channel = image.shape[1]
            if channel == self.config.vae_latent_channels:
                return image
                
            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn("Passing `image` as torch tensor with value range in [-1,1] is deprecated.", FutureWarning)
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image
    
    def postprocess(self, image: torch.Tensor, output_type: str = "pil", do_denormalize: Optional[List[bool]] = None) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor")
        
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (f"the output_type {output_type} is outdated and has been set to `np`.")
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if output_type == "latent":
            return image

        if not self.enable_async_postprocess or image.shape[0] <= 2:
            return self._postprocess_sync(image, output_type, do_denormalize)
        
        return self._postprocess_sync_optimized(image, output_type, do_denormalize)
    
    def postprocess_async(self, image: torch.Tensor, output_type: str = "pil", 
                         do_denormalize: Optional[List[bool]] = None,
                         callback: Optional[Callable] = None, request_id: str = "",
                         priority: int = 0) -> bool:
        if not self.enable_async_postprocess:
            raise RuntimeError("Async postprocessing is not enabled")
        
        return self.async_postprocessor.schedule_postprocessing(
            tensor=image,
            output_type=output_type,
            do_denormalize=do_denormalize,
            callback=callback,
            request_id=request_id,
            priority=priority
        )
    
    def _postprocess_sync(self, image: torch.Tensor, output_type: str, do_denormalize: Optional[List[bool]]) -> Any:
        image = self._denormalize_conditionally(image, do_denormalize)
        
        if output_type == "pt":
            return image
        
        image = self.pt_to_numpy(image)
        
        if output_type == "np":
            return image
        
        if output_type == "pil":
            return self.numpy_to_pil(image)
    
    def _postprocess_sync_optimized(self, image: torch.Tensor, output_type: str, do_denormalize: Optional[List[bool]]) -> Any:
        original_device = image.device
        if image.device.type == 'cuda':
            image = image.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        image = self._denormalize_conditionally(image, do_denormalize)
        
        if output_type == "pt":
            return image.to(original_device) if original_device.type == 'cuda' else image
        
        if output_type == "np":
            return self.pt_to_numpy(image)
        
        if output_type == "pil":
            numpy_images = self.pt_to_numpy(image)
            return self._numpy_to_pil_chunked(numpy_images)

    def apply_overlay(self, mask: PIL.Image.Image, init_image: PIL.Image.Image, image: PIL.Image.Image, crop_coords: Optional[Tuple[int, int, int, int]] = None) -> PIL.Image.Image:
        width, height = init_image.width, init_image.height

        init_image_masked = PIL.Image.new("RGBa", (width, height))
        init_image_masked.paste(init_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))
        init_image_masked = init_image_masked.convert("RGBA")

        if crop_coords is not None:
            x, y, x2, y2 = crop_coords
            w = x2 - x
            h = y2 - y
            base_image = PIL.Image.new("RGBA", (width, height))
            image = self.resize(image, height=h, width=w, resize_mode="crop")
            base_image.paste(image, (x, y))
            image = base_image.convert("RGB")

        image = image.convert("RGBA")
        image.alpha_composite(init_image_masked)
        return image.convert("RGB")

    
    def get_performance_stats(self) -> dict:
        stats = {}
        
        if hasattr(self, 'async_postprocessor'):
            stats['async_postprocessor'] = self.async_postprocessor.get_queue_status()
        
        if hasattr(self, 'memory_batcher'):
            stats['memory_batcher'] = {
                'last_batch_size': self.memory_batcher.last_batch_size
            }
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'async_postprocessor'):
            self.async_postprocessor.shutdown()
        
        if hasattr(self, 'pil_executor'):
            self.pil_executor.shutdown(wait=True)


class InpaintProcessor(ConfigMixin):

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(self, do_resize: bool = True, vae_scale_factor: int = 8, vae_latent_channels: int = 4, resample: str = "lanczos", reducing_gap: int = None, do_normalize: bool = True, do_binarize: bool = False, do_convert_grayscale: bool = False, mask_do_normalize: bool = False, mask_do_binarize: bool = True, mask_do_convert_grayscale: bool = True):
        super().__init__()

        self._image_processor = VaeImageProcessor(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            resample=resample,
            reducing_gap=reducing_gap,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )
        self._mask_processor = VaeImageProcessor(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            resample=resample,
            reducing_gap=reducing_gap,
            do_normalize=mask_do_normalize,
            do_binarize=mask_do_binarize,
            do_convert_grayscale=mask_do_convert_grayscale,
        )

    def preprocess(self, image: PIL.Image.Image, mask: PIL.Image.Image = None, height: int = None, width: int = None, padding_mask_crop: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None and padding_mask_crop is not None:
            raise ValueError("mask must be provided if padding_mask_crop is provided")

        if mask is None:
            return self._image_processor.preprocess(image, height=height, width=width)

        if padding_mask_crop is not None:
            crops_coords = self._image_processor.get_crop_region(mask, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        processed_image = self._image_processor.preprocess(image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode)
        processed_mask = self._mask_processor.preprocess(mask, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords)

        if crops_coords is not None:
            postprocessing_kwargs = {"crops_coords": crops_coords, "original_image": image, "original_mask": mask}
        else:
            postprocessing_kwargs = {"crops_coords": None, "original_image": None, "original_mask": None}

        return processed_image, processed_mask, postprocessing_kwargs

    def postprocess(self, image: torch.Tensor, output_type: str = "pil", original_image: Optional[PIL.Image.Image] = None, original_mask: Optional[PIL.Image.Image] = None, crops_coords: Optional[Tuple[int, int, int, int]] = None) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        image = self._image_processor.postprocess(image, output_type=output_type)

        if crops_coords is not None and (original_image is None or original_mask is None):
            raise ValueError("original_image and original_mask must be provided if crops_coords is provided")
        elif crops_coords is not None and output_type != "pil":
            raise ValueError("output_type must be 'pil' if crops_coords is provided")
        elif crops_coords is not None:
            image = [self._image_processor.apply_overlay(original_mask, original_image, i, crops_coords) for i in image]

        return image