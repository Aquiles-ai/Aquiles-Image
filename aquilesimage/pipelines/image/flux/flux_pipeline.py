import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from aquilesimage.utils import setup_colored_logger
import os
import logging
from typing import Literal
from aquilesimage.models import LoRAConfig
from aquilesimage.runtime import loadLoRA
from aquilesimage.models import BasePipeline

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineFlux(BasePipeline):
    def __init__(self, model_path: str | None = None, low_vram: bool = False, 
                load_lora: bool = False, conf_lora: LoRAConfig | None = None, 
                mode: Literal["eager", "piecewise"] = "eager" ):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram
        self.mode = mode
        self.pipelines = {}
        self.load_lora = load_lora
        self.conf_lora = conf_lora

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.info("Loading CUDA")

            self.device = "cuda"

            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)

            if self.load_lora:
                loadLoRA(self.pipeline, self.conf_lora)

            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass

            self.optimization()

                
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            
        else:
            raise Exception("No CUDA or MPS device available")

    def optimization(self):
        try:
            logger_p.info("Starting optimization process...")
            
            config = torch._inductor.config
            config.conv_1x1_as_mm = True
            config.coordinate_descent_check_all_directions = False
            config.coordinate_descent_tuning = False
            config.disable_progress = False
            config.epilogue_fusion = False
            config.shape_padding = True

            logger_p.info("Fusing QKV projections...")
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()

            logger_p.info("Converting to channels_last memory format...")
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)

            logger_p.info("FlashAttention")
            self.enable_flash_attn()

            if self.mode == "piecewise":
                self.pipeline.transformer = torch.compile(
                    self.pipeline.transformer, dynamic=False
                )
            
            logger_p.info("All optimizations completed successfully")
            
        except Exception as e:
            logger_p.error(f"X Error in optimization with Flux: {e}")
            raise

    ATTENTION_BACKEND_PRIORITY: tuple[str, ...] = ("_flash_3_hub", "flash", "sage_hub")

    def enable_flash_attn(self):
        for backend in self.ATTENTION_BACKEND_PRIORITY:
            if not self._attention_backend_ready(backend):
                logger_p.debug(f"Attention backend {backend} not available")
                continue
            try:
                self.pipeline.transformer.set_attention_backend(backend)
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

    def warmup_compile(self, batch_sizes, resolutions, prompt="warmup", steps=4):
        if self.mode != "piecewise":
            return
        for batch in batch_sizes:
            for h, w in resolutions:
                logger_p.info(f"Compiling shape batch={batch} {h}x{w}...")
                _ = self.pipeline(
                    prompt=[prompt] * batch,
                    height=h,
                    width=w,
                    num_inference_steps=steps,
                )
                logger_p.info(f"Shape batch={batch} {h}x{w} compiled and cached")