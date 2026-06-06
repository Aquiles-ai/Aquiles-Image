import torch
try:
    from diffusers.pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
    from diffusers.pipelines.ideogram4.prompt_enhancer import Ideogram4PromptEnhancerHead
    from diffusers.pipelines.ideogram4 import pipeline_ideogram4 as _ideogram4_module
except ImportError as e:
    print("Error import Ideogram4Pipeline")
    pass
from aquilesimage.utils import setup_colored_logger
import logging
from aquilesimage.models import LoRAConfig
from aquilesimage.runtime import loadLoRA
from aquilesimage.models import BasePipeline
import inspect

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

# This is used to override the class and always keep prompt upsampling active

class Ideogram4PipelineAlwaysUpsample(Ideogram4Pipeline):
    def __call__(self, *args, **kwargs):
        if args:
            sig = inspect.signature(Ideogram4Pipeline.__call__)
            params = list(sig.parameters.keys())[1:]  # skip 'self'
            for i, val in enumerate(args):
                if i < len(params):
                    kwargs[params[i]] = val
            args = ()

        kwargs.setdefault("prompt_upsampling", True)
        kwargs.pop("guidance_scale", None)
        kwargs.setdefault("guidance_schedule", (7.0,) * 45 + (3.0,) * 3)

        original = _ideogram4_module.is_outlines_available
        _ideogram4_module.is_outlines_available = lambda: False
        try:
            return super().__call__(**kwargs)
        finally:
            _ideogram4_module.is_outlines_available = original

class PipelineIdeogram4(BasePipeline):
    def __init__(self, model_path: str | None = None, dist_inf: bool = False,
                load_lora: bool = False, conf_lora: LoRAConfig | None = None):
        self.model_name = model_path
        try:
            self.pipeline: Ideogram4PipelineAlwaysUpsample | None = None
        except Exception as e:
            self.pipeline = None
            logger_p.info("Error import Ideogram4Pipeline")
            pass
        self.load_lora = load_lora
        self.conf_lora = conf_lora
        self.model_prompt = "diffusers/qwen3-vl-8b-instruct-lm-head"

    def start(self):
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        torch._inductor.config.max_autotune_gemm = True
        torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
        torch._inductor.config.triton.cudagraphs = False

        logger_p.info("Loading Ideogram4PromptEnhancerHead...")
        
        prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
            self.model_prompt,
            torch_dtype=torch.bfloat16,
        )

        logger_p.info("Loading Ideogram4PipelineAlwaysUpsample...")

        self.pipeline = Ideogram4PipelineAlwaysUpsample.from_pretrained(
            self.model_name,
            prompt_enhancer_head=prompt_enhancer_head,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

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
            logger_p.info("QKV projections fused")
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()
            logger_p.info("Channels last memory format enabled")
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)
            try:
                logger_p.info("FlashAttention")
                self.enable_flash_attn()
            except Exception as ea:
                logger_p.warning(f"Error in optimization (flash_attn): {str(ea)}")
                pass
        except Exception as e:
            logger_p.warning(f"Error in optimization: {str(e)}")
            pass
        