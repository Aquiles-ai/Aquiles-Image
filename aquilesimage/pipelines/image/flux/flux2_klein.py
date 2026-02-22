import torch
try:
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
except ImportError as e:
    print("Error import Flux2Pipeline")
    pass
from aquilesimage.utils import setup_colored_logger
import logging
from transformers import Qwen3ForCausalLM

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineFlux2Klein:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.model_name = model_path
        try:
            self.pipeline: Flux2KleinPipeline | None = None
        except Exception as e:
            self.pipeline = None
            print("Error import Flux2KleinPipeline")
            pass
        self.text_encoder: Qwen3ForCausalLM | None = None
        self.dit: Flux2Transformer2DModel | None = None
        self.vae: AutoencoderKLFlux2 | None
        self.device: str | None = None

    def start(self):
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        torch._inductor.config.max_autotune_gemm = True
        torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
        torch._inductor.config.triton.cudagraphs = False

        logger_p.info("Loading text encoder... (CUDA)")

        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
                self.model_name,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                device_map="cuda")

        logger_p.info("Loading Transformer... (CUDA)")
        self.dit = Flux2Transformer2DModel.from_pretrained(
                self.model_name, 
                subfolder="transformer", 
                torch_dtype=torch.bfloat16, 
                device_map="cuda")

        logger_p.info("Loading VAE... (CUDA)")
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.model_name,
            subfolder="vae",
            torch_dtype=torch.bfloat16).to("cuda")

        self.pipeline = Flux2KleinPipeline.from_pretrained(
            self.model_name, text_encoder=self.text_encoder, transformer=self.dit, vae=self.vae, dtype=torch.bfloat16
        ).to(device="cuda")

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