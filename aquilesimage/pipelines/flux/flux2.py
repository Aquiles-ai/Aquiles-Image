import torch
try:
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
except ImportError as e:
    print("Error import Flux2Pipeline")
    pass
from transformers import Mistral3ForConditionalGeneration
from aquilesimage.utils import setup_colored_logger
import os
import logging

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineFlux2:
    def __init__(self, model_path: str | None = None, low_vram: bool = False, device_map: str | None = None, dist_inf: bool = False):

        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.dist_inf = dist_inf
        if self.dist_inf and device_map == "cpu":
            raise ValueError("Distributed inference is only available for full CUDA loading; CPU loading cannot be used.")
        try:
            self.pipeline: Flux2Pipeline | None = None
        except Exception as e:
            self.pipeline = None
            print("Error import Flux2Pipeline")
            pass
        self.text_encoder: Mistral3ForConditionalGeneration | None = None
        self.dit: Flux2Transformer2DModel | None = None
        self.vae: AutoencoderKLFlux2 | None
        self.device: str | None = None
        self.low_vram = low_vram
        self.device_map = device_map
        self.pipelines = {}

    def start(self):
        if torch.cuda.is_available():
            if self.low_vram and self.device_map == 'cuda':
                self.start_low_vram_cuda()
            elif self.low_vram:
                self.start_low_vram()
            else:  
                logger_p.info(f"Loading FLUX.2 from {self.model_path}...")

                logger_p.info("Loading text encoder... (CUDA)")
                self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                    self.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda"
                )

                logger_p.info("Loading DiT transformer... (CUDA)")
                self.dit = Flux2Transformer2DModel.from_pretrained(
                    self.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda"
                )

                logger_p.info("Loading VAE... (CUDA)")
                self.vae = AutoencoderKLFlux2.from_pretrained(
                    self.model_path,
                    subfolder="vae",
                    torch_dtype=torch.bfloat16).to("cuda")

                logger_p.info("Converting all parameters to bfloat16...")
                self.dit = self.dit.to(torch.bfloat16)
                self.vae = self.vae.to(torch.bfloat16)


                logger_p.info("Creating FLUX.2 pipeline... (CUDA)")
                self.pipeline = Flux2Pipeline.from_pretrained(
                    self.model_path, text_encoder=self.text_encoder, transformer=self.dit, vae=self.vae, dtype=torch.bfloat16
                ).to(device="cuda")

                self.optimization()

    def start_low_vram(self):
        logger_p.info("Loading quantized text encoder...")
        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
        )

        logger_p.info("Loading quantized DiT transformer...")
        self.dit = AutoModel.from_pretrained(
            self.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda"
        )

        logger_p.info("Creating FLUX.2 pipeline...")
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.model_path, text_encoder=self.text_encoder, transformer=self.dit, torch_dtype=torch.bfloat16
        )

        logger_p.info("Enabling model CPU offload...")
        self.pipeline.enable_model_cpu_offload()

    def enable_flash_attn(self):
        if self.model_path == "black-forest-labs/FLUX.2-dev":
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
        else:
            logger_p.info("Skip FlashAttention")

    def optimization(self):
        try:
            if self.model_path == "black-forest-labs/FLUX.2-dev":
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

    def start_low_vram_cuda(self):
        logger_p.info("Loading quantized text encoder... (CUDA)")
        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_path, subfolder="text_encoder", dtype=torch.bfloat16, device_map="cuda"
        )

        logger_p.info("Loading quantized DiT transformer... (CUDA)")
        self.dit = AutoModel.from_pretrained(
            self.model_path, subfolder="transformer", device_map="cuda"
        )

        logger_p.info("Creating FLUX.2 pipeline... (CUDA)")
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.model_path, text_encoder=self.text_encoder, transformer=self.dit, dtype=torch.bfloat16
        ).to(device="cuda")

        self.optimization()