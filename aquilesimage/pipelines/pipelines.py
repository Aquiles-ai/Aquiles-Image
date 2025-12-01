from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
try:
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
except ImportError as e:
    print("Error import Flux2Pipeline")
    pass
try:
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
except ImportError as e:
    print("Error import ZImagePipeline")
    pass
from diffusers.models.auto_model import AutoModel
from transformers import Mistral3ForConditionalGeneration
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
import torch
import os
import logging
from aquilesimage.models import ImageModel
from aquilesimage.utils import setup_colored_logger


logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

"""
Maybe this will mutate with the changes implemented in diffusers
"""

class PipelineSD3:
    def __init__(self, model_path: str | None = None):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline | None = None
        self.device: str | None = None

    def start(self):
        torch.set_float32_matmul_precision("high")

        if hasattr(torch._inductor, 'config'):
            if hasattr(torch._inductor.config, 'conv_1x1_as_mm'):
                torch._inductor.config.conv_1x1_as_mm = True
            if hasattr(torch._inductor.config, 'coordinate_descent_tuning'):
                torch._inductor.config.coordinate_descent_tuning = True
            if hasattr(torch._inductor.config, 'epilogue_fusion'):
                torch._inductor.config.epilogue_fusion = False
            if hasattr(torch._inductor.config, 'coordinate_descent_check_all_directions'):
                torch._inductor.config.coordinate_descent_check_all_directions = True

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True

        if torch.cuda.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger_p.debug("Loading CUDA")
            self.device = "cuda"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)

            torch.cuda.empty_cache()

            if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                self.pipeline.transformer = self.pipeline.transformer.to(
                    memory_format=torch.channels_last
                )

            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("xformers enabled")
            except Exception as e:
                print("xformers not available:", e)

        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger_p.debug("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineFlux:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.debug("Loading CUDA")
            self.device = "cuda"

            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                ).to(device=self.device)

            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
                
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.debug("Loading MPS for Mac M Series")
            self.device = "mps"
            
        else:
            raise Exception("No hay dispositivo CUDA o MPS disponible")


class PipelineFluxKontext:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxKontextPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-Kontext-dev"
            logger_p.debug("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-Kontext-dev"
            logger_p.debug("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineFluxKontextMask:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxKontextPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-Kontext-dev"
            logger_p.debug("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-Kontext-dev"
            logger_p.debug("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineFlux2:
    def __init__(self, model_path: str | None = None, low_vram: bool = True):

        self.model_path = model_path or os.getenv("MODEL_PATH")
        try:
            self.pipeline: Flux2Pipeline | None = None
        except Exception as e:
            self.pipeline = None
            print("Error import Flux2Pipeline")
            pass
        self.text_encoder: Mistral3ForConditionalGeneration | None = None
        self.dit: AutoModel | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            if self.low_vram:
                self.start_low_vram()
        logger_p.debug(f"Loading FLUX.2 from {self.model_path}...")
        
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16
        )
        
        logger_p.debug("Enabling model CPU offload...")
        self.pipeline.enable_model_cpu_offload()


    def start_low_vram(self):
        logger_p.debug("Loading quantized text encoder...")
        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
        )

        logger_p.debug("Loading quantized DiT transformer...")
        self.dit = AutoModel.from_pretrained(
            self.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
        )

        logger_p.debug("Creating FLUX.2 pipeline...")
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.model_path, text_encoder=self.text_encoder, transformer=self.dit, torch_dtype=torch.bfloat16
        )

        logger_p.debug("Enabling model CPU offload...")
        self.pipeline.enable_model_cpu_offload()

    def enable_flash_attn(self):
        try:
            self.pipeline.transformer.set_attention_backend("_flash_3")
            logger_p.info("FLUX.2 - Flash Attention 3 enabled")
        except Exception as e:
            logger_p.debug(f"Flash Attention 3 not available: {str(e)}")
            try:
                self.pipeline.transformer.set_attention_backend("flash")
                logger_p.info("FLUX.2 - Flash Attention 2 enabled")
            except Exception as e2:
                logger_p.debug(f"Flash Attention 2 not available: {str(e2)}")
                logger_p.info("FLUX.2 - Using default attention backend (SDPA)")


class PipelineZImage:
    def __init__(self, model_path: str | None = None):

        self.model_path = model_path or os.getenv("MODEL_PATH")
        try:
            self.pipeline: ZImagePipeline | None = None
        except Exception as e:
            self.pipeline = None
            print("Error import ZImagePipeline")
            pass
        self.device: str | None = None

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "Tongyi-MAI/Z-Image-Turbo"
            logger_p.debug("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = ZImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to(device=self.device)

            self.pipeline.to("cuda")

            self.enable_flash_attn()

            if(self.compile_dit()):
                self._warmup()

    def enable_flash_attn(self):
        try:
            self.pipeline.transformer.set_attention_backend("flash")
            logger_p.info("Z-Image-Turbo - FlashAttention 2.0 is enabled")
            return True
        except Exception as e:
            logger_p.error(f"Z-Image-Turbo - FlashAttention 2.0 could not be enabled: {str(e)}")
            try:
                self.pipeline.transformer.set_attention_backend("_flash_3")
                logger_p.info("Z-Image-Turbo - FlashAttention 3.0 is enabled")
                return True
            except Exception as e3:
                logger_p.error(f"X Z-Image-Turbo - FlashAttention 3.0 could not be enabled: {str(e3)}")
            return False

    def compile_dit(self):
        try:
            self.pipeline.transformer.compile()
            logger_p.info("The DiT compilation is complete")
            return True
        except Exception as e:
            logger_p.error(f"Z-Image-Turbo - DiT could not be compiled: {str(e)}")
            return False

    def _warmup(self):
        try:
            logger_p.info("Starting warmup process...")
            warmup_prompt = "a simple test image"
            _ = self.pipeline(
                prompt=warmup_prompt,
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=torch.Generator(self.device).manual_seed(42),
            ).images[0]
        
            logger_p.info("Warmup completed successfully")
        
        except Exception as e:
            logger_p.error(f"X Warmup failed: {str(e)}")


class ModelPipelineInit:
    def __init__(self, model: str, low_vram: bool = False):
        self.model = model
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None
        self.low_vram = low_vram

        self.models = ImageModel

        self.stablediff3 = [
            self.models.SD3_MEDIUM,
            self.models.SD3_5_LARGE,
            self.models.SD3_5_LARGE_TURBO,
            self.models.SD3_5_MEDIUM
        ]

        self.flux = [
            self.models.FLUX_1_DEV,
            self.models.FLUX_1_SCHNELL,
            self.models.FLUX_1_KREA_DEV
        ]

        self.flux_kontext = [
            self.models.FLUX_1_KONTEXT_DEV
        ]

        self.z_image = [
            self.models.Z_IMAGE_TURBO
        ]

        self.flux2 = [
            self.models.FLUX_2_4BNB,
            self.models.FLUX_2
        ]


    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Base Models
        if self.model in self.stablediff3:
            self.pipeline = PipelineSD3(self.model)
        elif self.model in self.flux:
            self.pipeline = PipelineFlux(self.model, self.low_vram)
        elif self.model in self.z_image:
            self.pipeline = PipelineZImage(self.model)
        elif self.model in self.flux2:
            if self.model == 'diffusers/FLUX.2-dev-bnb-4bit':
                self.pipeline = PipelineFlux2(self.model, True)
            else:
                self.pipeline = PipelineFlux2(self.model, False)
        # Edition Models
        elif self.model in self.flux_kontext:
            self.pipeline = PipelineFluxKontext(self.model)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        return self.pipeline