import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
try:
    from diffusers.pipelines.glm_image.pipeline_glm_image import GlmImagePipeline
    from transformers import T5EncoderModel, ByT5Tokenizer, GlmImageProcessor, GlmImageForConditionalGeneration
    from diffusers.models.transformers.transformer_glm_image import GlmImageTransformer2DModel
except ImportError as e:
    print("Error import GlmImagePipeline")
    pass
from aquilesimage.utils import setup_colored_logger
import logging

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineGLMImage:
    def __init__(self, model_path: str):
        self.model_name = model_path

        self.pipeline: GlmImagePipeline | None = None
        self.text_encoder: T5EncoderModel | None = None
        self.vision_encoder: GlmImageForConditionalGeneration | None = None
        self.tokenizer: ByT5Tokenizer | None = None
        self.proccesor: GlmImageProcessor | None = None
        self.transformer: GlmImageTransformer2DModel | None = None
        self.vae: AutoencoderKL | None = None

    def start(self):
        if torch.cuda.is_available():
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
            torch._inductor.config.max_autotune_gemm = True
            torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
            torch._inductor.config.triton.cudagraphs = False

            self.text_encoder = T5EncoderModel.from_pretrained(self.model_name, 
                subfolder="text_encoder", device_map="cuda")

            self.vision_encoder = GlmImageForConditionalGeneration.from_pretrained(self.model_name, 
                subfolder="vision_language_encoder", device_map="cuda")

            self.tokenizer = ByT5Tokenizer.from_pretrained(self.model_name,
                subfolder="tokenizer")

            self.proccesor = GlmImageProcessor.from_pretrained(self.model_name,
                subfolder="processor", device_map="cuda")

            self.transformer = GlmImageTransformer2DModel.from_pretrained(self.model_name,
                subfolder="transformer", device_map="cuda")

            self.vae = AutoencoderKL.from_pretrained(self.model_name,
                subfolder="vae", device_map="cuda")

            self.pipeline = GlmImagePipeline.from_pretrained(self.model_name,
                text_encoder=self.text_encoder,
                vision_language_encoder=self.vision_encoder,
                tokenizer=self.tokenizer,
                processor=self.proccesor,
                transformer=self.transformer,
                vae=self.vae, device_map="cuda")

            # For now, only these optimizations are being applied, as GLM-Image has errors with FlashAttention.

            self.optimize_memory_format()
        
    def optimize_memory_format(self):
        try:
            logger_p.info("channels_last memory format")
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer.to(memory_format=torch.channels_last)
        except Exception as e:
            logger_p.error(f"X Error optimizing memory format: {e}")
            pass