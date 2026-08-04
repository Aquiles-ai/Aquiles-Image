import logging
from abc import ABC, abstractmethod

logger_p = logging.getLogger("Aquiles-Image-Pipelines")

class BasePipeline(ABC):

    ATTENTION_BACKEND_PRIORITY: tuple[str, ...] = ("_flash_3_hub", "flash", "sage_hub")

    def __init__(self, **kwargs):
        self.pipeline = None

    @abstractmethod
    def start(self):
        """Loads the model and prepares the pipeline for inference."""

    @abstractmethod
    def optimization(self):
        """Applies the optimizations specific to each pipeline."""

    def enable_flash_attn(self):
        if self.pipeline is None:
            logger_p.warning("No pipeline loaded, skipping flash attention")
            return

        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            logger_p.warning("No transformer component found for flash attention")
            return

        if not hasattr(transformer, "set_attention_backend"):
            logger_p.warning(
                "set_attention_backend not available for this model, skipping flash attention"
            )
            return

        for backend in self.ATTENTION_BACKEND_PRIORITY:
            if not self._attention_backend_ready(backend):
                logger_p.debug(f"Attention backend {backend} not available")
                continue
            try:
                transformer.set_attention_backend(backend)
                logger_p.info(f"Attention backend enabled: {backend}")
                return
            except Exception as e:
                logger_p.debug(f"Failed to set attention backend {backend}: {str(e)}")

        logger_p.warning("No optimized attention available, using default SDPA")

    def _attention_backend_ready(self, backend: str) -> bool:
        try:
            from diffusers.models.attention_dispatch import (
                AttentionBackendName,
                _HUB_KERNELS_REGISTRY,
                _check_attention_backend_requirements,
            )
            name = AttentionBackendName(backend)
            _check_attention_backend_requirements(name)
            if name in _HUB_KERNELS_REGISTRY:
                config = _HUB_KERNELS_REGISTRY[name]
                return self._hub_kernel_ready(config.repo_id, config.version)
            return True
        except Exception as e:
            logger_p.debug(f"Attention backend {backend} not usable: {str(e)}")
            return False

    def _hub_kernel_ready(self, repo_id: str, version: int | None) -> bool:
        try:
            from kernels import get_kernel, has_kernel
        except Exception as e:
            logger_p.debug(f"kernels package not usable: {str(e)}")
            return False
        try:
            if has_kernel(repo_id, version=version):
                return True
            get_kernel(repo_id, version=version)
            return True
        except Exception as e:
            logger_p.debug(f"Hub kernel {repo_id} not usable: {str(e)}")
            return False
