from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
try:
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
except ImportError as e:
    print("Error import Flux2Pipeline")
    pass
try:
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
except ImportError as e:
    print("Error import ZImagePipeline")
    pass
from diffusers.models.auto_model import AutoModel
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from transformers import Mistral3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
import torch
import os
import logging
from aquilesimage.models import ImageModel
from aquilesimage.utils import setup_colored_logger
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

"""
Maybe this will mutate with the changes implemented in diffusers
"""

class PipelineSD3:
    def __init__(self, model_path: str | None = None):
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
            logger_p.info("Loading CUDA")
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
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineFlux:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.info("Loading CUDA")
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
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            
        else:
            raise Exception("No hay dispositivo CUDA o MPS disponible")


class PipelineFluxKontext:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxKontextPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-Kontext-dev"
            logger_p.info("Loading CUDA")
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
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineFlux2:
    def __init__(self, model_path: str | None = None, low_vram: bool = True, device_map: str | None = None):

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
        self.device_map = device_map

    def start(self):
        if torch.cuda.is_available():
            if self.low_vram and self.device_map == 'cuda':
                self.start_low_vram_cuda()
            elif self.low_vram:
                self.start_low_vram()
            else:  
                logger_p.info(f"Loading FLUX.2 from {self.model_path}...")
        
                self.pipeline = Flux2Pipeline.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.bfloat16
                )
        
                logger_p.info("Enabling model CPU offload...")
                self.pipeline.enable_model_cpu_offload()


    def start_low_vram(self):
        logger_p.info("Loading quantized text encoder...")
        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
        )

        logger_p.info("Loading quantized DiT transformer...")
        self.dit = AutoModel.from_pretrained(
            self.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
        )

        logger_p.info("Creating FLUX.2 pipeline...")
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.model_path, text_encoder=self.text_encoder, transformer=self.dit, torch_dtype=torch.bfloat16
        )

        logger_p.info("Enabling model CPU offload...")
        self.pipeline.enable_model_cpu_offload()

    def enable_flash_attn(self):
        try:
            self.pipeline.transformer.set_attention_backend("_flash_3")
            logger_p.info("FLUX.2 - Flash Attention 3 enabled")
        except Exception as e:
            logger_p.info(f"Flash Attention 3 not available: {str(e)}")
            try:
                self.pipeline.transformer.set_attention_backend("flash")
                logger_p.info("FLUX.2 - Flash Attention 2 enabled")
            except Exception as e2:
                logger_p.info(f"Flash Attention 2 not available: {str(e2)}")
                logger_p.info("FLUX.2 - Using default attention backend (SDPA)")

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


class PipelineZImage:
    def __init__(self, model_path: str | None = None):

        self.model_path = model_path or os.getenv("MODEL_PATH")
        try:
            self.pipeline: ZImagePipeline | None = None
            self.transformer_z: ZImageTransformer2DModel | None = None
        except Exception as e:
            self.pipeline = None
            self.transformer_z = None
            print("Error import ZImagePipeline")
            pass
        self.device: str | None = None
        self.vae: AutoencoderKL | None = None
        self.text_encoder: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.scheduler: FlowMatchEulerDiscreteScheduler | None

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "Tongyi-MAI/Z-Image-Turbo"
            logger_p.info("Loading CUDA")
            self.device = "cuda"
            self.load_compo()
            self.pipeline = ZImagePipeline(
                scheduler=None,
                vae=self.vae,
                text_encoder=self.text_encoder, 
                tokenizer=self.tokenizer,
                transformer=None
            )
            
            self.pipeline.to("cuda")
            self.pipeline.vae.disable_tiling()
            self.load_transformer()
            self.enable_flash_attn()
            self.load_scheduler()

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

    def _warmup(self):
        try:
            logger_p.info("Starting warmup process...")
            warmup_prompt = "a simple test image"
            for i in range(3):
                _ = self.pipeline(
                    prompt=warmup_prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                    generator=torch.Generator(self.device).manual_seed(42 + i),
                ).images[0]      
            logger_p.info("Warmup completed successfully")
        except Exception as e:
            logger_p.error(f"X Warmup failed: {str(e)}")

    def load_compo(self):
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path or "Tongyi-MAI/Z-Image-Turbo",
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )

            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                self.model_path or "Tongyi-MAI/Z-Image-Turbo",
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path or "Tongyi-MAI/Z-Image-Turbo", 
                    subfolder="tokenizer")

            self.tokenizer.padding_side = "left"

            try:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.max_autotune_gemm = True
                torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
                torch._inductor.config.triton.cudagraphs = False

            except Exception as e:
                logger_p.error(f"X load_compo failed: {str(e)}")
                pass

        except Exception as e:
            logger_p.error(f"X load_compo failed: {str(e)}")

    def load_transformer(self):
        self.transformer = ZImageTransformer2DModel.from_pretrained(
            self.model_path or "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer").to("cuda", torch.bfloat16)
        self.pipeline.transformer = self.transformer

    def load_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        self.pipeline.scheduler = self.scheduler


class AutoPipelineDiffusers:
    def __init__(self, model_path: str | None = None):
        self.pipeline: AutoPipelineForText2Image | None = None
        self.model_name = model_path

    def start(self):
        if torch.cuda.is_available():
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.model_name, device_map="cuda")

class ModelPipelineInit:
    def __init__(self, model: str, low_vram: bool = False, auto_pipeline: bool = False, device_map_flux2: str | None = None):
        self.model = model
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None
        self.low_vram = low_vram
        self.auto_pipeline = auto_pipeline
        self.device_map_flux2 = device_map_flux2

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
                self.pipeline = PipelineFlux2(self.model, True, self.device_map_flux2)
            else:
                self.pipeline = PipelineFlux2(self.model, False)
        # Edition Models
        elif self.model in self.flux_kontext:
            self.pipeline = PipelineFluxKontext(self.model)
        elif self.auto_pipeline:
            logger_p.info(f"Loading model '{self.model}' with 'AutoPipelineDiffusers' - Experimental")
            self.pipeline = AutoPipelineDiffusers(self.model)
        else:
            raise ValueError(f"Unsupported model or enable the '--auto-pipeline' option (Only the Text2Image models). Model: {self.model}")

        return self.pipeline