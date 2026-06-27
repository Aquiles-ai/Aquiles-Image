import torch
from aquilesimage.utils import setup_colored_logger
import logging
from aquilesimage.models import LoRAConfig
from aquilesimage.runtime import loadLoRA
from aquilesimage.models import BasePipeline
import inspect

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

try:
    from diffusers import Krea2Pipeline
except ImportError as e:
    logger_p.info("Error import Krea2Pipeline")
    pass


class PipelineKrea2(BasePipeline):
    def __init__(self, model_path: str | None = None, dist_inf: bool = False,
                load_lora: bool = False, conf_lora: LoRAConfig | None = None):
        self.model_name = model_path

        try:
            self.pipeline: Krea2Pipeline | None = None
        except Exception as e:
            self.pipeline = None
            logger_p.info("Error import Krea2Pipeline")
            pass
        self.load_lora = load_lora
        self.conf_lora = conf_lora

    def start(self):
        from diffusers.quantizers import DiffusersAutoQuantizer

        original_from_config = DiffusersAutoQuantizer.from_config
        DiffusersAutoQuantizer.from_config = classmethod(lambda cls, *args, **kwargs: None)

        try:
            self.pipeline = Krea2Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        finally:
            DiffusersAutoQuantizer.from_config = original_from_config
        

        if self.load_lora:
            loadLoRA(self.pipeline, self.conf_lora)

        self.optimization()


    def enable_flash_attn(self):
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
                try:
                    self.pipeline.transformer.set_attention_backend("sage_hub")
                    logger_p.info("SAGE Attention enabled")
                except Exception as e3:
                    logger_p.warning(f"No optimized attention available, using default SDPA: {str(e3)}")

    def optimization(self):
        try:
            logger_p.info("Skip QKV projections fused & Channels last memory format enabled")
            #logger_p.info("QKV projections fused")
            #self.pipeline.transformer.fuse_qkv_projections()
            #self.pipeline.vae.fuse_qkv_projections()
            #logger_p.info("Channels last memory format enabled")
            #self.pipeline.transformer.to(memory_format=torch.channels_last)
            #self.pipeline.vae.to(memory_format=torch.channels_last)
            try:
                logger_p.info("FlashAttention")
                self.enable_flash_attn()
            except Exception as ea:
                logger_p.warning(f"Error in optimization (flash_attn): {str(ea)}")
                pass
        except Exception as e:
            logger_p.warning(f"Error in optimization: {str(e)}")
            pass
        