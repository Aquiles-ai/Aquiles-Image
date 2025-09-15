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
        if torch.cuda.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger_p.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class TextToImagePipelineFlux:
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