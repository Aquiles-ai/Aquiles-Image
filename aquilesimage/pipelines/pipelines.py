from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
import torch
import os
import logging
from aquilesimage.models import ImageModel
from aquilesimage.utils import setup_colored_logger
import gc



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
            logger_p.info(f"Loading CUDA with model: {model_path}")
            self.device = "cuda"
            
            torch.cuda.empty_cache()
            gc.collect()
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16" if "fp16" in model_path else None,
                low_cpu_mem_usage=True,
            )
            
            self.pipeline = self.pipeline.to(device=self.device)
            
            if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                self.pipeline.transformer = self.pipeline.transformer.to(
                    memory_format=torch.channels_last
                )
                logger_p.info("Transformer optimized with channels_last format")
            
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                self.pipeline.vae = self.pipeline.vae.to(
                    memory_format=torch.channels_last
                )
                logger_p.info("VAE optimized with channels_last format")
            
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger_p.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger_p.info(f"XFormers not available: {e}")
            
            if torch.__version__ >= "2.0.0":
                try:
                    if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                        self.pipeline.transformer = torch.compile(
                            self.pipeline.transformer,
                            mode="reduce-overhead",
                            fullgraph=False, 
                            dynamic=True,
                        )
                        logger_p.info("Transformer compiled with torch.compile")
                    
                    if hasattr(self.pipeline.vae, 'decode'):
                        self.pipeline.vae.decode = torch.compile(
                            self.pipeline.vae.decode,
                            mode="reduce-overhead",
                            fullgraph=False,
                        )
                        logger_p.info("VAE decoder compiled with torch.compile")
                        
                except Exception as e:
                    logger_p.warning(f"Could not compile components: {e}")
                    logger_p.info("Running without torch.compile optimization")
            
            self.pipeline.eval()
            for param in self.pipeline.parameters():
                param.requires_grad = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger_p.info("CUDA pipeline fully optimized and ready")
            
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger_p.info(f"Loading MPS for Mac M Series with model: {model_path}")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device=self.device)

            if hasattr(self.pipeline, 'transformer') and self.pipeline.transformer is not None:
                self.pipeline.transformer = self.pipeline.transformer.to(
                    memory_format=torch.channels_last
                )
            
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                self.pipeline.vae = self.pipeline.vae.to(
                    memory_format=torch.channels_last
                )
            
            self.pipeline.eval()
            for param in self.pipeline.parameters():
                param.requires_grad = False
                
            logger_p.info("MPS pipeline optimized and ready")
            
        else:
            raise Exception("No CUDA or MPS device available")
        
        # ============================================================
        # OPTIONAL WARMUP (Uncomment if you want to preheat)
        # ============================================================
        # self._warmup()
        
        logger_p.info("Pipeline initialization completed successfully")
    
    def _warmup(self):
        if self.pipeline:
            logger_p.info("Running warmup inference...")
            with torch.no_grad():
                _ = self.pipeline(
                    prompt="warmup",
                    num_inference_steps=1,
                    height=512,
                    width=512,
                    guidance_scale=1.0,
                )
            torch.cuda.empty_cache() if self.device == "cuda" else None
            logger_p.info("Warmup completed")

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
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

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
                torch_dtype=torch.bfloat16,
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
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineQwenImage:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: QwenImagePipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "Qwen/Qwen-Image"
            logger_p.info("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = QwenImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "Qwen/Qwen-Image"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = QwenImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class PipelineQwenImageEdit:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: QwenImageEditPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "Qwen/Qwen-Image"
            logger_p.info("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "Qwen/Qwen-Image"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")


class ModelPipelineInit:
    def __init__(self, model: str):
        self.model = model
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None

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

        self.qwen = [
            self.models.QWEN_IMAGE
        ]

        self.qwen_edit = [
            self.models.QWEN_IMAGE_EDIT
        ]

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Base Models
        if self.model in self.stablediff3:
            self.pipeline = PipelineSD3(self.model)
        elif self.model in self.flux:
            self.pipeline = PipelineFlux(self.model)
        elif self.model in self.qwen:
            self.pipeline = PipelineQwenImage(self.model)
        # Edition Models
        elif self.model in self.flux_kontext:
            self.pipeline = PipelineFluxKontext(self.model)
        elif self.model in self.qwen_edit:
            self.pipeline = PipelineQwenImageEdit(self.model)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        return self.pipeline