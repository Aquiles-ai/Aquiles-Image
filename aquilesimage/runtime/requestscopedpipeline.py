from typing import Optional, Any, Iterable, List, Dict
import copy
import threading
import torch
from diffusers.utils import logging
from aquilesimage.utils import setup_colored_logger
import logging
from .scheduler import BaseAsyncScheduler, async_retrieve_timesteps
from .wrappers import ThreadSafeTokenizerWrapper, ThreadSafeVAEWrapper, ThreadSafeImageProcessorWrapper

logger = setup_colored_logger("Aquiles-Image-Runtime-RequestScopedPipeline", logging.INFO)

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def _get_image_seq_len(height: int, width: int, patch_size: int = 16) -> int:
    return (height // patch_size) * (width // patch_size)

class RequestScopedPipeline:
    DEFAULT_MUTABLE_ATTRS = [
        "_all_hooks",
        "_offload_device", 
        "_progress_bar_config",
        "_progress_bar",
        "_rng_state",
        "_last_seed",
        "latents",
    ]

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        pipelines: Optional[Dict[str, Any]] = None,
        mutable_attrs: Optional[Iterable[str]] = None,
        auto_detect_mutables: bool = True,
        tensor_numel_threshold: int = 1_000_000,
        tokenizer_lock: Optional[threading.Lock] = None,
        wrap_scheduler: bool = True,
        use_flux: bool = False,
        use_kontext: bool = False,
        is_dist: bool = False,
    ):
        self.is_dist = is_dist

        if is_dist:
            if pipelines is None or len(pipelines) == 0:
                raise ValueError("pipelines dict cannot be None/empty in distributed mode")
            self.pipelines = pipelines
            self._base = None
            reference_pipeline = next(iter(pipelines.values()))
        else:
            if pipeline is None:
                raise ValueError("pipeline cannot be None in non-distributed mode")
            self._base = pipeline
            self.pipelines = {}
            reference_pipeline = pipeline

        self._is_auto_pipeline = 'AutoPipeline' in reference_pipeline.__class__.__name__

        if self._is_auto_pipeline:
            logger.info("AutoPipeline detected - limiting mutable attribute cloning")
    
        self.use_flux = use_flux
        if self.use_flux and hasattr(reference_pipeline, 'scheduler') and reference_pipeline.scheduler is not None:
            if hasattr(reference_pipeline.scheduler, 'config'):
                if hasattr(reference_pipeline.scheduler.config, 'use_dynamic_shifting'):
                    reference_pipeline.scheduler.config.use_dynamic_shifting = False

        self.unet = getattr(reference_pipeline, "unet", None)
        self.vae = getattr(reference_pipeline, "vae", None) 
        self.text_encoder = getattr(reference_pipeline, "text_encoder", None)
        self.components = getattr(reference_pipeline, "components", None)
        self.transformer = getattr(reference_pipeline, "transformer", None)

        self.is_kontext = use_kontext
        logger.info(f"is_kontext={self.is_kontext}, is_dist={self.is_dist}")

        if not is_dist:
            if wrap_scheduler and hasattr(reference_pipeline, 'scheduler') and reference_pipeline.scheduler is not None:
                if not isinstance(reference_pipeline.scheduler, BaseAsyncScheduler):
                    reference_pipeline.scheduler = BaseAsyncScheduler(reference_pipeline.scheduler)

        self._mutable_attrs = list(mutable_attrs) if mutable_attrs is not None else list(self.DEFAULT_MUTABLE_ATTRS)
        self._tokenizer_lock = tokenizer_lock if tokenizer_lock is not None else threading.Lock()
        self._vae_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._auto_detect_mutables = bool(auto_detect_mutables)
        self._tensor_numel_threshold = int(tensor_numel_threshold)
        self._auto_detected_attrs: List[str] = []



    def _make_local_scheduler(self, num_inference_steps: int, device: Optional[str] = None, **clone_kwargs):
        base_sched = getattr(self._base, "scheduler", None)
        if base_sched is None:
            return None

        if not isinstance(base_sched, BaseAsyncScheduler):
            wrapped_scheduler = BaseAsyncScheduler(base_sched)
        else:
            wrapped_scheduler = base_sched

        try:
            return wrapped_scheduler.clone_for_request(
                num_inference_steps=num_inference_steps, 
                device=device, 
                **clone_kwargs
            )
        except Exception as e:
            logger.info(f"clone_for_request failed: {e}; trying shallow copy fallback")
            try:
                if hasattr(wrapped_scheduler, 'scheduler'):
                    try:
                        copied_scheduler = copy.copy(wrapped_scheduler.scheduler)
                        return BaseAsyncScheduler(copied_scheduler)
                    except Exception:
                        return wrapped_scheduler
                else:
                    copied_scheduler = copy.copy(wrapped_scheduler)
                    if self.use_flux and hasattr(copied_scheduler, 'config'):
                        if hasattr(copied_scheduler.config, 'use_dynamic_shifting'):
                            copied_scheduler.config.use_dynamic_shifting = False
                    return BaseAsyncScheduler(copied_scheduler)
            except Exception as e2:
                logger.warning(f"Shallow copy of scheduler also failed: {e2}. Using original scheduler (*thread-unsafe but functional*).")
                return wrapped_scheduler 


    def _autodetect_mutables_from_pipeline(self, pipeline: Any, max_attrs: int = 40) -> List[str]:
        if not self._auto_detect_mutables:
            return []

        if self._auto_detected_attrs:
            return self._auto_detected_attrs

        candidates: List[str] = []
        seen = set()
    
        for name in dir(pipeline):
            if name.startswith("__"):
                continue
            if name in self._mutable_attrs:
                continue
            if name in ("to", "save_pretrained", "from_pretrained"):
                continue
            
            try:
                val = getattr(pipeline, name)
            except Exception:
                continue

            import types

            if callable(val) or isinstance(val, (types.ModuleType, types.FunctionType, types.MethodType)):
                continue

            if isinstance(val, (dict, list, set, tuple, bytearray)):
                candidates.append(name)
                seen.add(name)
            else:
                try:
                    if isinstance(val, torch.Tensor):
                        if val.numel() <= self._tensor_numel_threshold:
                            candidates.append(name)
                            seen.add(name)
                        else:
                            logger.info(f"Ignoring large tensor attr '{name}', numel={val.numel()}")
                except Exception:
                    continue

            if len(candidates) >= max_attrs:
                break

        self._auto_detected_attrs = candidates
        logger.info(f"Autodetected mutable attrs: {self._auto_detected_attrs}")
        return self._auto_detected_attrs

    def _is_readonly_property(self, base_obj, attr_name: str) -> bool:
        try:
            cls = type(base_obj)
            descriptor = getattr(cls, attr_name, None)
            if isinstance(descriptor, property):
                return descriptor.fset is None
            if hasattr(descriptor, "__set__") is False and descriptor is not None:
                return False
        except Exception:
            pass
        return False

    def _is_frozen_dict_or_config(self, val) -> bool:
        if val is None:
            return False
        class_name = val.__class__.__name__
        if 'FrozenDict' in class_name or 'frozendict' in class_name.lower():
            return True
    
        if hasattr(val, '_internal_dict'):
            return True

        if hasattr(val, '__class__') and hasattr(val.__class__, '__module__'):
            module_name = val.__class__.__module__
            if 'diffusers' in module_name and hasattr(val, 'to_dict'):
                return True
    
        return False

    def _clone_mutable_attrs(self, base, local, reference_pipeline=None):
        if reference_pipeline is None:
            reference_pipeline = base
        attrs_to_clone = list(self._mutable_attrs)
        attrs_to_clone.extend(self._autodetect_mutables_from_pipeline(reference_pipeline))

        if self._is_auto_pipeline:
            EXCLUDE_ATTRS = {
                "components",
                "config",                   
                "_internal_dict",            
                "model_index",               
                "_execution_device",        
                "_cached_hidden_states",     
                "_name_or_path",             
                "_class_name",              
                "_diffusers_version",       
            }
            logger.info("Using safe cloning strategy for generic pipeline")
        else:
            EXCLUDE_ATTRS = {"components"}

        for attr in attrs_to_clone:
            if attr in EXCLUDE_ATTRS:
                logger.info(f"Skipping excluded attr '{attr}'")
                continue
            if not hasattr(base, attr):
                continue
            if self._is_readonly_property(base, attr):
                logger.info(f"Skipping read-only property '{attr}'")
                continue

            try:
                val = getattr(base, attr)
            except Exception as e:
                logger.info(f"Could not getattr('{attr}') on base pipeline: {e}")
                continue

            try:
                if self._is_frozen_dict_or_config(val):
                    logger.info(f"Sharing FrozenDict/ConfigMixin object '{attr}' (type: {val.__class__.__name__})")
                    setattr(local, attr, val)
                    continue
            
                if isinstance(val, dict):
                    setattr(local, attr, dict(val))
                elif isinstance(val, (list, tuple, set)):
                    setattr(local, attr, list(val))
                elif isinstance(val, bytearray):
                    setattr(local, attr, bytearray(val))
                else:
                    if isinstance(val, torch.Tensor):
                        if val.numel() <= self._tensor_numel_threshold:
                            setattr(local, attr, val.clone())
                        else:
                            setattr(local, attr, val)
                    else:
                        try:
                            setattr(local, attr, copy.copy(val))
                        except Exception:
                            setattr(local, attr, val)
            except (AttributeError, TypeError) as e:
                logger.info(f"Skipping cloning attribute '{attr}' because it is not settable: {e}")
                continue
            except Exception as e:
                logger.info(f"X Unexpected error cloning attribute '{attr}': {e}")
                continue

    def _is_tokenizer_component(self, component) -> bool:
        if component is None:
            return False
        
        tokenizer_methods = ['encode', 'decode', 'tokenize', '__call__']
        has_tokenizer_methods = any(hasattr(component, method) for method in tokenizer_methods)
        
        class_name = component.__class__.__name__.lower()
        has_tokenizer_in_name = 'tokenizer' in class_name
        
        tokenizer_attrs = ['vocab_size', 'pad_token', 'eos_token', 'bos_token']
        has_tokenizer_attrs = any(hasattr(component, attr) for attr in tokenizer_attrs)
        
        return has_tokenizer_methods and (has_tokenizer_in_name or has_tokenizer_attrs)

    def _should_wrap_tokenizers(self) -> bool:
        return True

    def _verify_pipeline_config(self, pipeline) -> bool:

        if not (self._is_auto_pipeline):
            return True
    
        try:
            if not hasattr(pipeline, 'config'):
                logger.warning("X Pipeline does not have 'config' attribute")
                return False
            
            config = pipeline.config

            if isinstance(config, dict) and not hasattr(config, '_internal_dict'):
                logger.warning(f"X config is a simple dict, should be FrozenDict or ConfigMixin")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"X Error verifying pipeline config: {e}")
            return False

    def _restore_config_if_needed(self, local_pipe):
        if not (self._is_auto_pipeline):
            return
    
        if not self._verify_pipeline_config(local_pipe):
            logger.warning("X Pipeline config corrupted after copy, attempting to restore...")
            if hasattr(self._base, '_internal_dict'):
                try:
                    local_pipe._internal_dict = self._base._internal_dict
                    logger.info("Config restored successfully")
                except Exception as e:
                    logger.warning(f"X Could not restore config: {e}")
        
            if hasattr(self._base, 'config') and hasattr(local_pipe, '_internal_dict'):
                try:
                    object.__setattr__(local_pipe, '_internal_dict', self._base._internal_dict)
                    logger.info("Config restored via object.__setattr__")
                except Exception as e:
                    logger.warning(f"X Alternative config restore failed: {e}")

    def _generate_batch_single(self, prompts: List[str], *args, num_inference_steps: int = 50, device: Optional[str] = None, **kwargs):
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        image_seq_len = _get_image_seq_len(height, width)
        
        mu = _calculate_shift(image_seq_len)

        if not prompts:
            raise ValueError("prompts list cannot be empty")
    
        if not isinstance(prompts, list):
            raise TypeError(f"prompts must be a list, got {type(prompts)}")
    
        if self.is_kontext:
            logger.debug(f"Kontext mode detected - calculating mu for resolution {height}x{width}")

            local_scheduler = self._make_local_scheduler(num_inference_steps=num_inference_steps, device=device, use_dynamic_shifting=True, mu=mu)
        elif self.use_flux:
            local_scheduler = self._make_local_scheduler(num_inference_steps=num_inference_steps, device=device, use_dynamic_shifting=False)
        else:
            local_scheduler = self._make_local_scheduler(num_inference_steps=num_inference_steps, device=device)

        try:
            local_pipe = copy.copy(self._base)
        except Exception as e:
            logger.warning(f"copy.copy(self._base) failed: {e}. Falling back to deepcopy (may increase memory).")
            local_pipe = copy.deepcopy(self._base)

        try:
            if hasattr(local_pipe, "vae") and local_pipe.vae is not None and not isinstance(local_pipe.vae, ThreadSafeVAEWrapper):
                local_pipe.vae = ThreadSafeVAEWrapper(local_pipe.vae, self._vae_lock)

            if hasattr(local_pipe, "image_processor") and local_pipe.image_processor is not None and not isinstance(local_pipe.image_processor, ThreadSafeImageProcessorWrapper):
                local_pipe.image_processor = ThreadSafeImageProcessorWrapper(local_pipe.image_processor, self._image_lock)
        except Exception as e:
            logger.info(f"Could not wrap vae/image_processor: {e}")

        self._restore_config_if_needed(local_pipe)

        if local_scheduler is not None:
            try:
                if self.is_kontext:
                    scheduler_kwargs = {k: v for k, v in kwargs.items() if k in ['timesteps', 'sigmas', 'mu']}
                    scheduler_kwargs['use_dynamic_shifting'] = True
                    scheduler_kwargs['mu'] = mu
                else:
                    scheduler_kwargs = {k: v for k, v in kwargs.items() if k in ['timesteps', 'sigmas']}
            
                timesteps, num_steps, configured_scheduler = async_retrieve_timesteps(
                    local_scheduler.scheduler,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    return_scheduler=True,
                    use_kontext=self.is_kontext,
                    **scheduler_kwargs
                )

                final_scheduler = BaseAsyncScheduler(configured_scheduler)
                setattr(local_pipe, "scheduler", final_scheduler)
            except Exception as e:
                logger.warning(f"Could not set scheduler on local pipe; proceeding without replacing scheduler. Error{e}")

        self._clone_mutable_attrs(self._base, local_pipe)

        num_images_per_prompt = kwargs.get('num_images_per_prompt', 1)
        logger.info(f"generate_batch - num_images_per_prompt:{num_images_per_prompt}")
        total_images = len(prompts) * num_images_per_prompt
        generators = []
        for _ in range(total_images):
            g = torch.Generator(device=device or "cuda")
            g.manual_seed(torch.randint(0, 10_000_000, (1,)).item())
            generators.append(g)

        original_tokenizers = {}
    
        if self._should_wrap_tokenizers():
            try:
                for name in dir(local_pipe):
                    if "tokenizer" in name and not name.startswith("_"):
                        tok = getattr(local_pipe, name, None)
                        if tok is not None and self._is_tokenizer_component(tok):
                            if not isinstance(tok, ThreadSafeTokenizerWrapper):
                                original_tokenizers[name] = tok
                                wrapped_tokenizer = ThreadSafeTokenizerWrapper(tok, self._tokenizer_lock)
                                setattr(local_pipe, name, wrapped_tokenizer)

                if hasattr(local_pipe, "components") and isinstance(local_pipe.components, dict):
                    for key, val in local_pipe.components.items():
                        if val is None:
                            continue
                    
                        if self._is_tokenizer_component(val):
                            if not isinstance(val, ThreadSafeTokenizerWrapper):
                                original_tokenizers[f"components[{key}]"] = val
                                wrapped_tokenizer = ThreadSafeTokenizerWrapper(val, self._tokenizer_lock)
                                local_pipe.components[key] = wrapped_tokenizer

            except Exception as e:
                logger.info(f"Tokenizer wrapping step encountered an error: {e}")

        result = None
        cm = getattr(local_pipe, "model_cpu_offload_context", None)
    
        try:

            if self.is_kontext:
                logger.info(f"Calling Kontext pipeline with mu={kwargs.get('mu')}")

            kwargs.pop('mu', None)
        
            if callable(cm):
                try:
                    with cm():
                        result = local_pipe(prompt=prompts,
                generator=generators, num_inference_steps=num_inference_steps, **kwargs)
                except TypeError:
                    try:
                        with cm:
                            result = local_pipe(prompt=prompts,
                generator=generators, num_inference_steps=num_inference_steps, **kwargs)
                    except Exception as e:
                        logger.info(f"model_cpu_offload_context usage failed: {e}. Proceeding without it.")
                        result = local_pipe(prompt=prompts,
                generator=generators, num_inference_steps=num_inference_steps, **kwargs)
            else:
                result = local_pipe(prompt=prompts,
                generator=generators, num_inference_steps=num_inference_steps, **kwargs)

            if len(result.images) != total_images:
                raise RuntimeError(
                    f"X CRITICAL: Pipeline returned {len(result.images)} images "
                    f"but expected {total_images} ({len(prompts)} prompts × {num_images_per_prompt} images each). "
                    f"Output mapping is BROKEN!"
                )

            logger.info(f"Batch of {len(prompts)} completed successfully")

            return result

        finally:
            try:
                for name, tok in original_tokenizers.items():
                    if name.startswith("components["):
                        key = name[len("components["):-1]
                        if hasattr(local_pipe, 'components') and isinstance(local_pipe.components, dict):
                            local_pipe.components[key] = tok
                    else:
                        setattr(local_pipe, name, tok)
            except Exception as e:
                logger.info(f"Error restoring original tokenizers: {e}")

    def _generate_batch_dist_(self, prompts: List[str], *args, num_inference_steps: int = 50, device: Optional[str] = None, **kwargs):
        # I believe that isolation is not needed in distributed inference, because queue management is already robust.
        if device is None:
            raise ValueError("device parameter is required in distributed mode")
    
        if device not in self.pipelines:
            raise ValueError(
                f"Device '{device}' not found in available pipelines. "
                f"Available: {list(self.pipelines.keys())}"
            )
    
        if not prompts:
            raise ValueError("prompts list cannot be empty")

        if not isinstance(prompts, list):
            raise TypeError(f"prompts must be a list, got {type(prompts)}")

        image = kwargs.pop("image", None)
        height = kwargs.pop("height", 1024)
        width = kwargs.pop("width", 1024)
        num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)

        base_pipeline = self.pipelines[device]
    
        logger.info(f"[DIST] Using pipeline on device {device} for {len(prompts)} prompts")

        logger.info(f"[DIST] generate_batch on {device} - num_images_per_prompt:{num_images_per_prompt}")
        total_images = len(prompts) * num_images_per_prompt
        generators = []
        for _ in range(total_images):
            g = torch.Generator(device=device)
            g.manual_seed(torch.randint(0, 10_000_000, (1,)).item())
            generators.append(g)

        result = None
        cm = getattr(base_pipeline, "model_cpu_offload_context", None)
        
        try:
            if callable(cm):
                try:
                    with cm():
                        if image is not None:
                            result = base_pipeline(
                                prompt=prompts,
                                generator=generators,
                                num_inference_steps=num_inference_steps,
                                image=image,
                                height=height,
                                width=width,
                                num_images_per_prompt=num_images_per_prompt,
                            )
                        else:
                            result = base_pipeline(
                                prompt=prompts,
                                generator=generators,
                                num_inference_steps=num_inference_steps,
                                height=height,
                                width=width,
                                num_images_per_prompt=num_images_per_prompt,
                            )
                except TypeError:
                    try:
                        with cm:
                            if image is not None:
                                result = base_pipeline(
                                    prompt=prompts,
                                    generator=generators,
                                    num_inference_steps=num_inference_steps,
                                    image=image,
                                    height=height,
                                    width=width,
                                    num_images_per_prompt=num_images_per_prompt,
                                )
                            else:
                                result = base_pipeline(
                                    prompt=prompts,
                                    generator=generators,
                                    num_inference_steps=num_inference_steps,
                                    height=height,
                                    width=width,
                                    num_images_per_prompt=num_images_per_prompt,
                                )
                    except Exception as e:
                        logger.info(f"model_cpu_offload_context failed: {e}. Proceeding without it.")
                        if image is not None:
                            result = base_pipeline(
                                prompt=prompts,
                                generator=generators,
                                num_inference_steps=num_inference_steps,
                                image=image,
                                height=height,
                                width=width,
                                num_images_per_prompt=num_images_per_prompt,
                            )
                        else:
                            result = base_pipeline(
                                prompt=prompts,
                                generator=generators,
                                num_inference_steps=num_inference_steps,
                                height=height,
                                width=width,
                                num_images_per_prompt=num_images_per_prompt,
                            )
            else:
                if image is not None:
                    result = base_pipeline(
                        prompt=prompts,
                        generator=generators,
                        num_inference_steps=num_inference_steps,
                        image=image,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images_per_prompt,
                    )
                else:
                    result = base_pipeline(
                        prompt=prompts,
                        generator=generators,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images_per_prompt,
                    )

            if len(result.images) != total_images:
                raise RuntimeError(
                    f"X CRITICAL: Pipeline returned {len(result.images)} images "
                    f"but expected {total_images} ({len(prompts)} prompts × {num_images_per_prompt} images each)."
                )

            logger.info(f"[DIST] Batch of {len(prompts)} completed on {device}")

            return result

        except Exception as e:
            logger.info(f"Error _generate_batch_dist_: {e}")

    def generate_batch(self, prompts: List[str], *args, num_inference_steps: int = 50, device: Optional[str] = None, **kwargs):
        if self.is_dist:
            return self._generate_batch_dist_(prompts, *args, num_inference_steps=num_inference_steps, device=device, **kwargs)
        else:
            return self._generate_batch_single(prompts, *args, num_inference_steps=num_inference_steps, device=device, **kwargs)