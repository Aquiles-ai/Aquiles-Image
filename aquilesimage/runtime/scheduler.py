from typing import Any, Optional, Union, List
import torch
import copy
import inspect
from diffusers.utils import logging

logger = logging.get_logger(__name__)

class BaseAsyncScheduler:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler

    def __getattr__(self, name: str):
        if hasattr(self.scheduler, name):
            return getattr(self.scheduler, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value):
        if name == 'scheduler':
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'scheduler') and hasattr(self.scheduler, name):
                setattr(self.scheduler, name, value)
            else:
                super().__setattr__(name, value)

    def clone_for_request(self, num_inference_steps: int, device: Union[str, torch.device, None] = None, **kwargs):
        try:
            local = copy.copy(self.scheduler)
            local.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
            cloned = self.__class__(local)
            return cloned
        except Exception as e1:
            logger.info(f"Error cloning scheduler 'e1': {e1}")
            try:
                scheduler_class = self.scheduler.__class__
                if hasattr(self.scheduler, 'config'):
                    local = scheduler_class.from_config(self.scheduler.config)
                else:
                    local = self.scheduler
                local.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
                cloned = self.__class__(local)
                return cloned
            except Exception as e2:
                logger.info(f"Error in the fallback when cloning the scheduler 'e2': {e2}")
                return self

    def __repr__(self):
        return f"BaseAsyncScheduler({repr(self.scheduler)})"
    
    def __str__(self):
        return f"BaseAsyncScheduler wrapping: {str(self.scheduler)}"


def async_retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.
    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Backwards compatible: by default the function behaves exactly as before and returns
        (timesteps_tensor, num_inference_steps)

    If the caller passes `return_scheduler=True` in kwargs, the function will **not** mutate the passed
    scheduler. Instead it will use a cloned scheduler if available (via `scheduler.clone_for_request`)
    or a fallback, call `set_timesteps` on that cloned scheduler, and return:
        (timesteps_tensor, num_inference_steps, scheduler_in_use)
    """
    # pop our optional control kwarg (keeps compatibility)
    return_scheduler = bool(kwargs.pop("return_scheduler", False))

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")

    # choose scheduler to call set_timesteps on
    scheduler_in_use = scheduler
    if return_scheduler:
        # Do not mutate the provided scheduler: prefer to clone if possible
        if hasattr(scheduler, "clone_for_request"):
            try:
                # clone_for_request may accept num_inference_steps or other kwargs; be permissive
                scheduler_in_use = scheduler.clone_for_request(num_inference_steps=num_inference_steps or 0, device=device)
            except Exception as e1:
                logger.info(f"Error cloning the scheduler with clone_for_request on 'async_retrieve_timesteps': {e1}")
                try:
                    scheduler_in_use = copy.copy(scheduler)
                except Exception as e2:
                    scheduler_in_use = scheduler
                    logger.info(f"Error cloning the scheduler in the fallback of clone_for_request in 'async_retrieve_timesteps': {e2}")
        else:
            try:
                scheduler_in_use = copy.copy(scheduler)
            except Exception as e3:
                logger.info(f"Error cloning the scheduler in 'async_retrieve_timesteps': {e3}")
                scheduler_in_use = scheduler


    # helper to test if set_timesteps supports a particular kwarg
    def _accepts(param_name: str) -> bool:
        try:
            return param_name in set(inspect.signature(scheduler_in_use.set_timesteps).parameters.keys())
        except (ValueError, TypeError):
            # if signature introspection fails, be permissive and attempt the call later
            return False

    # now call set_timesteps on the chosen scheduler_in_use (may be original or clone)
    if timesteps is not None:
        accepts_timesteps = _accepts("timesteps")
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler_in_use.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler_in_use.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps_out = scheduler_in_use.timesteps
        num_inference_steps = len(timesteps_out)
    elif sigmas is not None:
        accept_sigmas = _accepts("sigmas")
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler_in_use.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler_in_use.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps_out = scheduler_in_use.timesteps
        num_inference_steps = len(timesteps_out)
    else:
        # default path
        scheduler_in_use.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps_out = scheduler_in_use.timesteps

    if return_scheduler:
        return timesteps_out, num_inference_steps, scheduler_in_use
    return timesteps_out, num_inference_steps