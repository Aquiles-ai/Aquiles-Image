import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
from functools import lru_cache

import numpy as np
import torch

from aquilesimage.kernels.configs import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.utils.import_utils import is_scipy_available
from diffusers.utils.outputs import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin

if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass 
class TimestepCacheKey:
    num_inference_steps: int
    shift: float
    mu: Optional[float]
    use_dynamic_shifting: bool
    use_karras_sigmas: bool
    use_exponential_sigmas: bool  
    use_beta_sigmas: bool
    time_shift_type: str
    shift_terminal: Optional[float]
    invert_sigmas: bool
    num_train_timesteps: int
    sigmas_tuple: Optional[Tuple[float, ...]] = None
    timesteps_tuple: Optional[Tuple[float, ...]] = None
    
    def __hash__(self):
        return hash((
            self.num_inference_steps, self.shift, self.mu, 
            self.use_dynamic_shifting, self.use_karras_sigmas,
            self.use_exponential_sigmas, self.use_beta_sigmas,
            self.time_shift_type, self.shift_terminal, self.invert_sigmas,
            self.num_train_timesteps, self.sigmas_tuple, self.timesteps_tuple
        ))


@dataclass
class CachedTimesteps:
    timesteps: np.ndarray
    sigmas: np.ndarray
    device: str = "cpu"
    dtype: str = "float32"


class TimestepCache:    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[TimestepCacheKey, CachedTimesteps] = {}
        self.access_order: List[TimestepCacheKey] = []
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: TimestepCacheKey) -> Optional[CachedTimesteps]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: TimestepCacheKey, value: CachedTimesteps):
        """Cache computed timesteps"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate
            }
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0


class ThreadSafeSchedulerState:    
    def __init__(self, original_scheduler):
        self.original_config = original_scheduler.config
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None
        self._step_index = None
        self._begin_index = None
        
        self.sigma_min = original_scheduler.sigma_min
        self.sigma_max = original_scheduler.sigma_max
        
    def copy(self):
        new_state = ThreadSafeSchedulerState.__new__(ThreadSafeSchedulerState)
        new_state.original_config = self.original_config
        new_state.timesteps = self.timesteps
        new_state.sigmas = self.sigmas  
        new_state.num_inference_steps = self.num_inference_steps
        new_state._step_index = None
        new_state._begin_index = None
        new_state.sigma_min = self.sigma_min
        new_state.sigma_max = self.sigma_max
        return new_state


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1
    
    # Class-level cache shared across all instances
    _global_cache = TimestepCache(max_size=200)
    _cache_lock = threading.RLock()

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        enable_cache: bool = True,
        cache_size: int = 50,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError("Only one of beta/exponential/karras sigmas can be used.")
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # Store in thread-safe state container
        self._thread_state = ThreadSafeSchedulerState(self)
        self._thread_state.timesteps = sigmas * num_train_timesteps
        self._thread_state.sigmas = sigmas.to("cpu")
        
        self._shift = shift
        
        # Enhanced features
        self.enable_cache = enable_cache
        if enable_cache:
            # Instance-level cache for this scheduler
            self._local_cache = TimestepCache(max_size=cache_size)
    
    
    @property
    def shift(self):
        return self._shift
    
    @property 
    def step_index(self):
        return getattr(self._current_state, '_step_index', None) if hasattr(self, '_current_state') else None
    
    @property
    def begin_index(self):
        return getattr(self._current_state, '_begin_index', None) if hasattr(self, '_current_state') else None
    
    @property
    def timesteps(self):
        return getattr(self._current_state, 'timesteps', self._thread_state.timesteps) if hasattr(self, '_current_state') else self._thread_state.timesteps
    
    @property
    def sigmas(self):  
        return getattr(self._current_state, 'sigmas', self._thread_state.sigmas) if hasattr(self, '_current_state') else self._thread_state.sigmas
    
    @property
    def sigma_min(self):
        return self._thread_state.sigma_min
    
    @property
    def sigma_max(self):
        return self._thread_state.sigma_max
    
    
    def _create_cache_key(
        self, 
        num_inference_steps: Optional[int], 
        sigmas: Optional[List[float]], 
        mu: Optional[float],
        timesteps: Optional[List[float]]
    ) -> TimestepCacheKey:
        """Create cache key for timestep computation"""
        return TimestepCacheKey(
            num_inference_steps=num_inference_steps or 0,
            shift=self._shift,
            mu=mu,
            use_dynamic_shifting=self.config.use_dynamic_shifting,
            use_karras_sigmas=self.config.use_karras_sigmas,
            use_exponential_sigmas=self.config.use_exponential_sigmas,
            use_beta_sigmas=self.config.use_beta_sigmas,
            time_shift_type=self.config.time_shift_type,
            shift_terminal=self.config.shift_terminal,
            invert_sigmas=self.config.invert_sigmas,
            num_train_timesteps=self.config.num_train_timesteps,
            sigmas_tuple=tuple(sigmas) if sigmas else None,
            timesteps_tuple=tuple(timesteps) if timesteps else None
        )
    
    def _get_cached_timesteps(self, cache_key: TimestepCacheKey) -> Optional[CachedTimesteps]:
        if not self.enable_cache:
            return None
        
        cached = self._local_cache.get(cache_key)
        if cached is None:
            with self._cache_lock:
                cached = self._global_cache.get(cache_key)
                if cached:
                    self._local_cache.put(cache_key, cached)
        
        return cached
    
    def _cache_timesteps(self, cache_key: TimestepCacheKey, timesteps: np.ndarray, sigmas: np.ndarray):
        if not self.enable_cache:
            return
        
        cached = CachedTimesteps(
            timesteps=timesteps.copy(),
            sigmas=sigmas.copy()
        )
        
        self._local_cache.put(cache_key, cached)
        with self._cache_lock:
            self._global_cache.put(cache_key, cached)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_time_shift_exponential(mu: float, sigma: float, t_tuple: Tuple[float, ...]) -> Tuple[float, ...]:
        t = np.array(t_tuple)
        result = np.exp(mu) / (np.exp(mu) + (1 / t - 1) ** sigma)
        return tuple(result)
    
    @staticmethod  
    @lru_cache(maxsize=128)
    def _compute_time_shift_linear(mu: float, sigma: float, t_tuple: Tuple[float, ...]) -> Tuple[float, ...]:
        t = np.array(t_tuple)
        result = mu / (mu + (1 / t - 1) ** sigma)
        return tuple(result)
    
    def time_shift(self, mu: float, sigma: float, t: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(t, torch.Tensor):
            t_np = t.cpu().numpy()
            is_tensor = True
        else:
            t_np = t
            is_tensor = False
        
        t_tuple = tuple(t_np.flatten())
        
        if self.config.time_shift_type == "exponential":
            result_tuple = self._compute_time_shift_exponential(mu, sigma, t_tuple)
        elif self.config.time_shift_type == "linear":
            result_tuple = self._compute_time_shift_linear(mu, sigma, t_tuple)
        else:
            raise ValueError(f"Unknown time_shift_type: {self.config.time_shift_type}")
        
        result = np.array(result_tuple).reshape(t_np.shape)
        
        if is_tensor:
            return torch.from_numpy(result).to(t.device, t.dtype)
        return result
    
    
    def set_begin_index(self, begin_index: int = 0):
        if hasattr(self, '_current_state'):
            self._current_state._begin_index = begin_index
        else:
            self._current_state = self._thread_state.copy()
            self._current_state._begin_index = begin_index
    
    def set_shift(self, shift: float):
        self._shift = shift
    
    def get_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to True")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError("Length mismatch between sigmas/timesteps and num_inference_steps")
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        # Try cache first
        cache_key = self._create_cache_key(num_inference_steps, sigmas, mu, timesteps)
        cached_result = self._get_cached_timesteps(cache_key)
        
        if cached_result is not None:
            device = device or "cpu"
            cached_timesteps = torch.from_numpy(cached_result.timesteps).to(device=device, dtype=torch.float32)
            cached_sigmas = torch.from_numpy(cached_result.sigmas).to(device=device, dtype=torch.float32)
            return cached_timesteps, cached_sigmas, num_inference_steps
        
        is_timesteps_provided = timesteps is not None
        
        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), 
                    self._sigma_to_t(self.sigma_min), 
                    num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # Apply time shifting
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self._shift * sigmas / (1 + (self._shift - 1) * sigmas)

        # Apply terminal shifting if needed
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(torch.from_numpy(sigmas)).numpy()

        # Apply sigma schedule transformations
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(torch.from_numpy(sigmas), num_inference_steps).numpy()
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(torch.from_numpy(sigmas), num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(torch.from_numpy(sigmas), num_inference_steps)

        # Generate timesteps if not provided
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps

        # Handle sigma inversion for specific models (e.g., Mochi)
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            # Append terminal sigma
            sigmas = np.concatenate([sigmas, [1.0]])
        else:
            # Append terminal sigma
            sigmas = np.concatenate([sigmas, [0.0]])

        # Cache the result
        self._cache_timesteps(cache_key, timesteps, sigmas)

        # Convert to tensors and move to device  
        device = device or "cpu"
        timesteps_tensor = torch.from_numpy(timesteps).to(device=device, dtype=torch.float32)
        sigmas_tensor = torch.from_numpy(sigmas).to(device=device, dtype=torch.float32)

        return timesteps_tensor, sigmas_tensor, num_inference_steps
    
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,  
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        computed_timesteps, computed_sigmas, computed_steps = self.get_timesteps(
            num_inference_steps, device, sigmas, mu, timesteps
        )
        
        if not hasattr(self, '_current_state'):
            self._current_state = self._thread_state.copy()
        
        self._current_state.timesteps = computed_timesteps
        self._current_state.sigmas = computed_sigmas
        self._current_state.num_inference_steps = computed_steps
        self._current_state._step_index = None
        self._current_state._begin_index = None
    
    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        current_sigmas = self.sigmas
        current_timesteps = self.timesteps
        
        target_device = sample.device
        target_dtype = sample.dtype
        
        if current_sigmas.device != target_device or current_sigmas.dtype != target_dtype:
            # Cache device/dtype converted tensors to avoid repeated conversions
            cache_key = f"sigmas_{target_device}_{target_dtype}"
            if not hasattr(self, '_device_cache'):
                self._device_cache = {}
            
            if cache_key not in self._device_cache:
                self._device_cache[cache_key] = current_sigmas.to(device=target_device, dtype=target_dtype)
            sigmas = self._device_cache[cache_key]
        else:
            sigmas = current_sigmas

        # Handle timestep conversion efficiently
        if isinstance(timestep, torch.Tensor):
            if timestep.device != target_device:
                timestep = timestep.to(target_device)
            if sample.device.type == "mps" and torch.is_floating_point(timestep):
                schedule_timesteps = current_timesteps.to(target_device, dtype=torch.float32)
                timestep = timestep.to(target_device, dtype=torch.float32)
            else:
                schedule_timesteps = current_timesteps.to(target_device)
        else:
            schedule_timesteps = current_timesteps.to(target_device)
            timestep = torch.tensor([timestep], device=target_device, dtype=target_dtype)

        # Get step indices
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        return sigma * noise + (1.0 - sigma) * sample
    
    
    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        if isinstance(timestep, torch.Tensor) and timestep.numel() == 1:
            timestep = timestep.item()
        
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if not hasattr(self, '_current_state'):
            self._current_state = self._thread_state.copy()
        
        if self._current_state._begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._current_state._step_index = self.index_for_timestep(timestep)
        else:
            self._current_state._step_index = self._current_state._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError("Passing integer indices as timesteps is not supported.")

        if not hasattr(self, '_current_state'):
            self._current_state = self._thread_state.copy()
            
        if self._current_state._step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps
            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self._current_state._step_index
            sigma = self.sigmas[sigma_idx]  
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if self.config.stochastic_sampling:
            x0 = sample - current_sigma * model_output
            noise = torch.randn_like(sample, generator=generator)
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            prev_sample = sample + dt * model_output

        self._current_state._step_index += 1
        
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
    
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.from_numpy(sigmas)

    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> np.ndarray:
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6) -> np.ndarray:
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array([
            sigma_min + (ppf * (sigma_max - sigma_min))
            for ppf in [
                scipy.stats.beta.ppf(timestep, alpha, beta)
                for timestep in 1 - np.linspace(0, 1, num_inference_steps)
            ]
        ])
        return sigmas
    
    def get_cache_stats(self) -> Dict[str, Any]:
        stats = {"caching_enabled": self.enable_cache}
        
        if self.enable_cache:
            stats["local_cache"] = self._local_cache.get_stats()
            with self._cache_lock:
                stats["global_cache"] = self._global_cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        if self.enable_cache:
            self._local_cache.clear()
            with self._cache_lock:
                self._global_cache.clear()
        
        # Clear device cache
        if hasattr(self, '_device_cache'):
            self._device_cache.clear()
    
    def create_request_scheduler(self):
        new_scheduler = self.__class__.__new__(self.__class__)
        new_scheduler.config = self.config
        new_scheduler._shift = self._shift
        new_scheduler.enable_cache = self.enable_cache
        
        if self.enable_cache:
            new_scheduler._local_cache = TimestepCache(max_size=self._local_cache.max_size)
        
        new_scheduler._thread_state = self._thread_state.copy()
        
        return new_scheduler

    def __len__(self):
        return self.config.num_train_timesteps

    def cleanup_request_state(self):
        """Clean up per-request state to prevent memory leaks"""
        if hasattr(self, '_current_state'):
            delattr(self, '_current_state')
        
        if hasattr(self, '_device_cache'):
            self._device_cache.clear()