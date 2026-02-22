import torch
import logging
from aquilesimage.models import ImageModel
from aquilesimage.utils import setup_colored_logger
from aquilesimage.pipelines.image.stable_diff_3_5 import PipelineSD3
from aquilesimage.pipelines.image.flux import PipelineFlux, PipelineFlux2Klein, PipelineFlux2, PipelineFluxKontext
from aquilesimage.pipelines.image.z_image import PipelineZImageTurbo, PipelineZImage
from aquilesimage.pipelines.image.qwen_image import PipelineQwenImage, PipelineQwenImageEdit
from aquilesimage.pipelines.image.glm import PipelineGLMImage
from aquilesimage.pipelines.image.auto import AutoPipelineDiffusers

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

## TODO: Implement the AutoPipeline for Img2Img.

class ModelPipelineInit:
    def __init__(self, model: str, low_vram: bool = False, auto_pipeline: bool = False, device_map_flux2: str | None = None, dist_inf: bool = False):
        self.model = model
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None
        self.low_vram = low_vram
        self.auto_pipeline = auto_pipeline
        self.device_map_flux2 = device_map_flux2
        self.dist_inf = dist_inf

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

        self.z_image_base = [
            self.models.Z_IMAGE_BASE
        ]

        self.qwen_image = [
            self.models.QWEN_IMAGE,
            self.models.QWEN_IMAGE_2512
        ]

        self.qwen_image_edit = [
            self.models.QWEN_IMAGE_EDIT_BASE,
            self.models.QWEN_IMAGE_EDIT_2511,
            self.models.QWEN_IMAGE_EDIT_2509
        ]

        self.flux2 = [
            self.models.FLUX_2_4BNB,
            self.models.FLUX_2
        ]

        self.flux2_klein = [
            self.models.FLUX_2_KLEIN_4B, 
            self.models.FLUX_2_KLEIN_9B
        ]

        self.glm_image = [
            self.models.GLM
        ]


    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Base Models
        if self.model in self.stablediff3:
            self.pipeline = PipelineSD3(self.model, self.dist_inf)
        elif self.model in self.flux:
            self.pipeline = PipelineFlux(self.model, self.low_vram, False, self.dist_inf)
        elif self.model in self.z_image:
            self.pipeline = PipelineZImageTurbo(self.model, self.dist_inf)
        elif self.model in self.flux2:
            if self.model == 'diffusers/FLUX.2-dev-bnb-4bit':
                self.pipeline = PipelineFlux2(self.model, True, self.device_map_flux2, self.dist_inf)
            else:
                self.pipeline = PipelineFlux2(self.model, False, None, self.dist_inf)
        elif self.model in self.qwen_image:
            self.pipeline = PipelineQwenImage(self.model, self.dist_inf)
        elif self.model in self.qwen_image_edit:
            self.pipeline = PipelineQwenImageEdit(self.model, self.dist_inf)
        elif self.model in self.flux_kontext:
            self.pipeline = PipelineFluxKontext(self.model, self.low_vram, self.dist_inf)
        elif self.model in self.flux2_klein:
            self.pipeline = PipelineFlux2Klein(self.model, self.dist_inf)
        elif self.model in self.glm_image:
            self.pipeline = PipelineGLMImage(self.model)
        elif self.model in self.z_image_base:
            self.pipeline = PipelineZImage(self.model)
        elif self.auto_pipeline:
            logger_p.info(f"Loading model '{self.model}' with 'AutoPipelineDiffusers' - Experimental")
            self.pipeline = AutoPipelineDiffusers(self.model, self.dist_inf)
        else:
            raise ValueError(f"Unsupported model or enable the '--auto-pipeline' option (Only the Text2Image models). Model: {self.model}")

        return self.pipeline