import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from aquilesimage.utils import setup_colored_logger
import logging

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineQwenImage:
    def __init__(self, model_path: str | None, dist_inf: bool = False):
        self.pipeline: QwenImagePipeline | None = None
        self.model_name = model_path
        self.pipelines = {}
        self.dist_inf = dist_inf

    def start(self):
        if torch.cuda.is_available():
            self.pipeline = QwenImagePipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16
            ).to("cuda")
            self.optimization()
        else:
            raise ValueError("CUDA not available")

    def optimization(self):
        try:
            try:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.max_autotune_gemm = True
                torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
                torch._inductor.config.triton.cudagraphs = False
            except Exception as e:
                logger_p.error(f"X torch_opt failed: {str(e)}")
                pass
            self.flash_attn()
            self.fuse_qkv_projections()
            self.optimize_memory_format()
        except Exception as e:
            logger_p.error(f"X The optimizations could not be applied: {e}")
            logger_p.info("Running with the non-optimized version")
            pass

    def optimize_memory_format(self):
        try:
            logger_p.info("channels_last memory format")
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer.to(memory_format=torch.channels_last)
        except Exception as e:
            logger_p.error(f"X Error optimizing memory format: {e}")
            pass

    def fuse_qkv_projections(self):
        try:
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()
            logger_p.info("QKV projection fusion")
        except Exception as e:
            logger_p.error(f"X Error merging QKV projections: {e}")
            pass

    def flash_attn(self):
        try:
            self.pipeline.transformer.set_attention_backend("_flash_3_hub")
            logger_p.info("FlashAttention 3 enabled")
        except Exception as e:
            logger_p.debug(f"FlashAttention 3 not available: {str(e)}")
            try:
                self.pipeline.transformer.set_attention_backend("flash")
                logger_p.info("FlashAttention 2 enabled")
            except Exception as e2:
                logger_p.debug(f"FlashAttention 2 not available: {str(e2)}")
                pass