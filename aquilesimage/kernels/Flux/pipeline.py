import threading
import time
import gc
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from aquilesimage.kernels.Flux.image_processor import PipelineImageInput, VaeImageProcessor
from aquilesimage.kernels.Flux.schedulers import FlowMatchEulerDiscreteScheduler
from aquilesimage.kernels.models.transformersflux import FluxTransformer2DModel
from aquilesimage.runtime.scheduler import async_retrieve_timesteps as retrieve_timesteps
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.loaders.ip_adapter import FluxIPAdapterMixin
from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
from diffusers.loaders.single_file import FromSingleFileMixin
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.utils import logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.import_utils import is_torch_xla_available
from diffusers.utils.doc_utils import replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate dynamic shift value for Flux scheduling"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@dataclass
class RequestContext:
    """Context for tracking request-specific state and optimizations"""
    request_id: str = ""
    priority: int = 0
    start_time: float = field(default_factory=time.time)
    enable_optimizations: bool = True
    async_postprocess: bool = False
    callback_on_complete: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"flux_req_{int(time.time() * 1000)}"


class MemoryManager:    
    def __init__(self, target_memory_usage: float = 0.85):
        self.target_memory_usage = target_memory_usage
        self.last_cleanup = 0
        self.cleanup_interval = 30  # seconds
        self.lock = threading.RLock()
        
    def should_cleanup(self) -> bool:
        current_time = time.time()
        with self.lock:
            if current_time - self.last_cleanup < self.cleanup_interval:
                return False
                
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if reserved > 0:
                usage_ratio = allocated / reserved
                return usage_ratio > self.target_memory_usage
        
        return False
    
    def cleanup_memory(self, force: bool = False):
        current_time = time.time()
        with self.lock:
            if not force and current_time - self.last_cleanup < self.cleanup_interval:
                return
                
            self.last_cleanup = current_time
        
        # Garbage collection
        gc.collect()
        
        # CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        stats = {}
        if torch.cuda.is_available():
            stats = {
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            }
        return stats


class TextEncodingCache:    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _create_key(self, prompts: List[str], max_sequence_length: int, 
                   num_images_per_prompt: int) -> str:
        """Create cache key from prompt parameters"""
        prompt_str = "|".join(sorted(prompts)) if isinstance(prompts, list) else prompts
        return f"{hash(prompt_str)}_{max_sequence_length}_{num_images_per_prompt}"
    
    def get(self, prompts: List[str], max_sequence_length: int, 
            num_images_per_prompt: int) -> Optional[Dict[str, torch.Tensor]]:
        key = self._create_key(prompts, max_sequence_length, num_images_per_prompt)
        
        with self.lock:
            if key in self.cache:
                # Move to end (LRU)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hit_count += 1
                return self.cache[key].copy()  # Return copy to prevent mutation
            else:
                self.miss_count += 1
                return None
    
    def put(self, prompts: List[str], max_sequence_length: int, 
            num_images_per_prompt: int, encodings: Dict[str, torch.Tensor]):
        key = self._create_key(prompts, max_sequence_length, num_images_per_prompt)
        
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove oldest
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            # Store copies to prevent external mutation
            self.cache[key] = {k: v.clone() for k, v in encodings.items()}
            self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hit_count + self.miss_count
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': self.hit_count / total if total > 0 else 0.0
            }
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0


class FluxPipelineKernels(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    r"""
    Optimized Flux pipeline for text-to-image generation.


    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            CLIP text encoder for prompt encoding.
        text_encoder_2 ([`T5EncoderModel`]):
            T5 text encoder for enhanced prompt encoding.
        tokenizer (`CLIPTokenizer`):
            Tokenizer for CLIP text encoder.
        tokenizer_2 (`T5TokenizerFast`):
            Tokenizer for T5 text encoder.
        image_encoder ([`CLIPVisionModelWithProjection`], *optional*):
            Vision encoder for IP-Adapter support.
        feature_extractor ([`CLIPImageProcessor`], *optional*):
            Feature extractor for image processing.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        enable_optimizations: bool = True,
        enable_text_encoding_cache: bool = True,
        text_cache_size: int = 500,
        enable_memory_management: bool = True,
        target_memory_usage: float = 0.85,
        enable_async_postprocess: bool = True,
        max_concurrent_postprocess: int = 3,
    ):
        super().__init__()

        # Register all components
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        
        # Calculate VAE scale factor 
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        
        # Initialize image processor with optimizations
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            enable_async_postprocess=enable_async_postprocess,
            max_concurrent_postprocess=max_concurrent_postprocess,
            enable_memory_aware_batching=True,
            enable_threaded_pil_ops=True,
        )
        
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

        self.enable_optimizations = enable_optimizations
        self.enable_async_postprocess = enable_async_postprocess
        
        # Initialize optimization components
        if enable_text_encoding_cache:
            self.text_encoding_cache = TextEncodingCache(max_size=text_cache_size)
        else:
            self.text_encoding_cache = None
            
        if enable_memory_management:
            self.memory_manager = MemoryManager(target_memory_usage=target_memory_usage)
        else:
            self.memory_manager = None
        
        # Thread-safe request tracking
        self._active_requests = {}
        self._request_lock = threading.RLock()
        
        # Performance statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_inference_time': 0.0,
            'cache_stats': {},
        }
        self.stats_lock = threading.RLock()

    def _create_request_context(self, **kwargs) -> RequestContext:
        """Create context for current request"""
        defaults = {
            'enable_optimizations': self.enable_optimizations,
            'async_postprocess': self.enable_async_postprocess,
        }
        defaults.update(kwargs)  
        return RequestContext(**defaults)

    def _register_request(self, context: RequestContext):
        """Register active request for tracking"""
        with self._request_lock:
            self._active_requests[context.request_id] = {
                'context': context,
                'start_time': context.start_time,
            }

    def _unregister_request(self, context: RequestContext, success: bool = True):
        """Unregister completed request and update stats"""
        with self._request_lock:
            if context.request_id in self._active_requests:
                del self._active_requests[context.request_id]
        
        with self.stats_lock:
            self.stats['total_requests'] += 1
            if success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            # Update average inference time
            inference_time = time.time() - context.start_time
            current_avg = self.stats['avg_inference_time']
            total = self.stats['total_requests']
            self.stats['avg_inference_time'] = (current_avg * (total - 1) + inference_time) / total

    def _get_cached_text_encodings(self, prompt: List[str], prompt_2: List[str], 
                                  max_sequence_length: int, num_images_per_prompt: int,
                                  device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached text encodings if available"""
        if not self.text_encoding_cache:
            return None
        
        # Create combined cache key
        combined_prompts = prompt + prompt_2 if prompt_2 else prompt
        cached = self.text_encoding_cache.get(
            combined_prompts, max_sequence_length, num_images_per_prompt
        )
        
        if cached:
            # Move tensors to correct device
            return {k: v.to(device) for k, v in cached.items()}
        
        return None

    def _cache_text_encodings(self, prompt: List[str], prompt_2: List[str],
                             max_sequence_length: int, num_images_per_prompt: int,
                             encodings: Dict[str, torch.Tensor]):
        """Cache text encodings for future use"""
        if not self.text_encoding_cache:
            return
        
        # Move to CPU for caching
        cpu_encodings = {k: v.cpu() for k, v in encodings.items()}
        combined_prompts = prompt + prompt_2 if prompt_2 else prompt
        
        self.text_encoding_cache.put(
            combined_prompts, max_sequence_length, num_images_per_prompt, cpu_encodings
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    
        transformer_path = pretrained_model_name_or_path
    
        kwargs['transformer'] = FluxTransformer2DModel.from_pretrained(
            transformer_path, 
            subfolder="transformer",
            torch_dtype=kwargs.get('torch_dtype', None),
            **{k: v for k, v in kwargs.items() if k.startswith('transformer_')}
        )
    
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        # Optimized tokenization
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # Check for truncation
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # Generate embeddings
        with torch.no_grad():
            prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # Duplicate embeddings for multiple images per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        # Optimized tokenization
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        # Generate embeddings
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # Duplicate embeddings for multiple images per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # Handle LoRA scaling
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # Try to get cached encodings
            cached_encodings = self._get_cached_text_encodings(
                prompt, prompt_2, max_sequence_length, num_images_per_prompt, device
            )
            
            if cached_encodings:
                prompt_embeds = cached_encodings['prompt_embeds']
                pooled_prompt_embeds = cached_encodings['pooled_prompt_embeds']
                text_ids = cached_encodings['text_ids']
            else:
                # Generate new encodings
                pooled_prompt_embeds = self._get_clip_prompt_embeds(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                )
                prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=prompt_2,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                )
                
                # Generate text_ids
                dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
                
                # Cache the results
                self._cache_text_encodings(
                    prompt, prompt_2, max_sequence_length, num_images_per_prompt,
                    {
                        'prompt_embeds': prompt_embeds,
                        'pooled_prompt_embeds': pooled_prompt_embeds,
                        'text_ids': text_ids
                    }
                )

        # Clean up LoRA scaling
        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        # Generate text_ids if not already created
        if 'text_ids' not in locals():
            dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def encode_image(self, image, device, num_images_per_prompt):
        if self.image_encoder is None:
            raise ValueError("Image encoder is required for IP-Adapter functionality")
            
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            image_embeds = self.image_encoder(image).image_embeds
        
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt
    ):
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. "
                    f"Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image, device, 1)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if len(ip_adapter_image_embeds) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. "
                    f"Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. "
                f"Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Prompt validation
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please make sure to only forward one of the two.")
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt_2` and `prompt_embeds`. Please make sure to only forward one of the two.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        # Negative prompt validation
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt_2` and `negative_prompt_embeds`.")

        # Embedding validation
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError("If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed.")
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError("If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed.")

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents


    def enable_vae_slicing(self):
        deprecate("enable_vae_slicing", "0.40.0", 
                 f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated. Please use `pipe.vae.enable_slicing()`.")
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        deprecate("disable_vae_slicing", "0.40.0",
                 f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated. Please use `pipe.vae.disable_slicing()`.")
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        deprecate("enable_vae_tiling", "0.40.0",
                 f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated. Please use `pipe.vae.enable_tiling()`.")
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        deprecate("disable_vae_tiling", "0.40.0",
                 f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated. Please use `pipe.vae.disable_tiling()`.")
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}. "
                f"Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def get_optimization_stats(self) -> Dict[str, Any]:
        stats = {}
        
        # Pipeline stats
        with self.stats_lock:
            stats['pipeline'] = self.stats.copy()
        
        # Text encoding cache stats
        if self.text_encoding_cache:
            stats['text_cache'] = self.text_encoding_cache.get_stats()
        
        # Image processor stats
        if hasattr(self.image_processor, 'get_performance_stats'):
            stats['image_processor'] = self.image_processor.get_performance_stats()
        
        # Scheduler cache stats
        if hasattr(self.scheduler, 'get_cache_stats'):
            stats['scheduler'] = self.scheduler.get_cache_stats()
        
        # Memory stats
        if self.memory_manager:
            stats['memory'] = self.memory_manager.get_memory_stats()
        
        # Active requests
        with self._request_lock:
            stats['active_requests'] = len(self._active_requests)
        
        return stats

    def clear_caches(self):
        if self.text_encoding_cache:
            self.text_encoding_cache.clear()
        
        if hasattr(self.image_processor, 'clear_caches'):
            self.image_processor.clear_caches()
        
        if hasattr(self.scheduler, 'clear_cache'):
            self.scheduler.clear_cache()
        
        if hasattr(self.transformer, 'clear_caches'):
            self.transformer.clear_caches()
        
        if self.memory_manager:
            self.memory_manager.cleanup_memory(force=True)

    def optimize_for_inference(self):
        if hasattr(self.transformer, 'optimize_for_inference'):
            self.transformer.optimize_for_inference()
        
        if hasattr(self.image_processor, 'optimize_for_inference'):
            self.image_processor.optimize_for_inference()
        
        if hasattr(self.scheduler, 'optimize_for_inference'):
            self.scheduler.optimize_for_inference()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        request_id: str = "",
        enable_optimizations: Optional[bool] = None,
        callback_on_complete: Optional[Callable] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        
        context = self._create_request_context(
            request_id=request_id,
            enable_optimizations=enable_optimizations if enable_optimizations is not None else self.enable_optimizations,
            callback_on_complete=callback_on_complete
        )
        
        self._register_request(context)
        
        try:
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            self.check_inputs(
                prompt, prompt_2, height, width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

            # Set instance variables
            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            lora_scale = (
                self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            )
            
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
            )
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
            
            (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            
            if do_true_cfg:
                (negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids) = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

            # Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels // 4
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
                sigmas = None
                
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
                negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
            ):
                negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
                negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
            ):
                ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}

            image_embeds = None
            negative_image_embeds = None
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image, ip_adapter_image_embeds, device, batch_size * num_images_per_prompt,
                )
            if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
                negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                    negative_ip_adapter_image, negative_ip_adapter_image_embeds, device, batch_size * num_images_per_prompt,
                )


            self.scheduler.set_begin_index(0)
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    
                    # Set IP-Adapter embeddings
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                    
                    # Broadcast timestep
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # Forward pass with caching
                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                    # True CFG if enabled
                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred = self.transformer(
                                hidden_states=latents,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                pooled_projections=negative_pooled_prompt_embeds,
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # Scheduler step
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            latents = latents.to(latents_dtype)

                    # Callback handling
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # Update progress
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            self._current_timestep = None

            if output_type == "latent":
                image = latents
            else:
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                
                # VAE decode
                with torch.no_grad():
                    image = self.vae.decode(latents, return_dict=False)[0]
                
                if context.enable_optimizations and hasattr(self.image_processor, 'postprocess_async'):
                    def completion_callback(result, req_id, error=None):
                        if error:
                            logger.error(f"Async postprocessing failed for {req_id}: {error}")
                        elif context.callback_on_complete:
                            context.callback_on_complete(result, req_id)
                    
                    success = self.image_processor.postprocess_async(
                        image, output_type=output_type,
                        callback=completion_callback,
                        request_id=context.request_id
                    )
                    
                    if not success:
                        image = self.image_processor.postprocess(image, output_type=output_type)
                else:
                    image = self.image_processor.postprocess(image, output_type=output_type)

            # Memory cleanup
            if self.memory_manager and self.memory_manager.should_cleanup():
                self.memory_manager.cleanup_memory()

            # Offload models
            self.maybe_free_model_hooks()

            # Mark request as successful
            self._unregister_request(context, success=True)

            # Handle callback
            if context.callback_on_complete and not (hasattr(self.image_processor, 'postprocess_async') and 
                                                   context.enable_optimizations):
                context.callback_on_complete(image, context.request_id)

            if not return_dict:
                return (image,)

            return FluxPipelineOutput(images=image)

        except Exception as e:
            # Mark request as failed and cleanup
            self._unregister_request(context, success=False)
            logger.error(f"Pipeline execution failed for request {context.request_id}: {e}")
            raise e