from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
from aquilesimage.utils import setup_colored_logger
import logging

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class AutoPipelineDiffusers:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.pipeline: AutoPipelineForText2Image | None = None
        self.model_name = model_path
        self.dist_inf = dist_inf
        self.pipelines = {}
        
    def start(self):
        if torch.cuda.is_available():
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.model_name, device_map="cuda")
            self.optimization()

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
            self.enable_flash_attn()
            self.optimize_memory_format()
            self.fuse_qkv_projections()
        except Exception as e:
            logger_p.error(f"X The optimizations could not be applied: {e}")
            logger_p.info("Running with the non-optimized version")
            pass

    def enable_flash_attn(self):
        component = None
        if hasattr(self.pipeline, 'transformer'):
            component = self.pipeline.transformer

        if component is None:
            logger_p.warning("No transformer component found for flash attention")
            return

        if not hasattr(component, 'set_attention_backend'):
            logger_p.warning("set_attention_backend not available for this model, skipping flash attention")
            return

        backends = [
            ("_flash_3_hub", "FlashAttention 3"),
            ("flash",        "FlashAttention 2"),
            ("sage_hub",     "SAGE Attention"),
        ]

        for backend, name in backends:
            try:
                component.set_attention_backend(backend)
                logger_p.info(f"{name} enabled")
                return
            except Exception as e:
                logger_p.debug(f"{name} not available: {str(e)}")

        logger_p.warning("No optimized attention backend available, using default SDPA")

    def optimize_memory_format(self): 
        try:
            logger_p.info("channels_last memory format")
            if hasattr(self.pipeline, 'unet'):
                self.pipeline.unet.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer.to(memory_format=torch.channels_last)
        except Exception as e:
            logger_p.error(f"X Error optimizing memory format: {e}")
            pass

    def fuse_qkv_projections(self):        
        try:
            self.pipeline.fuse_qkv_projections()
            logger_p.info("QKV projection fusion")
        except AttributeError:
            logger_p.warning("fuse_qkv_projections not available for this model")
            pass
        except Exception as e:
            logger_p.error(f"X Error merging QKV projections: {e}")
            pass
