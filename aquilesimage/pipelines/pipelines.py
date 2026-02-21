try:
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
except ImportError as e:
    print("Error import ZImagePipeline")
    pass
from diffusers.models.auto_model import AutoModel
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from transformers import Mistral3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
try:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
except ImportError as e:
    print("Error import QwenImageEditPlusPipeline")
    pass
import torch
import os
import logging
from aquilesimage.models import ImageModel
from aquilesimage.utils import setup_colored_logger
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import gc
try:
    from diffusers.pipelines.glm_image.pipeline_glm_image import GlmImagePipeline
    from transformers import T5EncoderModel, ByT5Tokenizer, GlmImageProcessor, GlmImageForConditionalGeneration
    from diffusers.models.transformers.transformer_glm_image import GlmImageTransformer2DModel
except ImportError as e:
    print("Error import GlmImagePipeline")
    pass
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from aquilesimage.pipelines.stable_diff_3_5 import PipelineSD3
from aquilesimage.pipelines.flux import PipelineFlux, PipelineFlux2Klein, PipelineFlux2, PipelineFluxKontext


logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineZImageTurbo:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):

        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.dist_inf = dist_inf
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
        self.pipelines = {}

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
            logger_p.error(f"X Z-Image-Turbo - FlashAttention 2.0 could not be enabled: {str(e)}")
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

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

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
                logger_p.error(f"X load_compo config failed: {str(e)}")
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


class PipelineQwenImage:
    def __init__(self, model_path: str | None, dist_inf: bool = False):
        self.pipeline: QwenImagePipeline | None = None
        self.model_name = model_path
        self.pipelines = {}
        self.dist_inf = dist_inf

    def start(self):
        if torch.cuda.is_available():
            self.pipeline = QwenImagePipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16
            ).to("cuda")
            self.optimization()
        else:
            raise ValueError("CUDA not available")

    def optimization(self):
        try:
            try:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.max_autotune_gemm = True
                torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
                torch._inductor.config.triton.cudagraphs = False
            except Exception as e:
                logger_p.error(f"X torch_opt failed: {str(e)}")
                pass
            self.flash_attn()
            self.fuse_qkv_projections()
            self.optimize_memory_format()
        except Exception as e:
            logger_p.error(f"X The optimizations could not be applied: {e}")
            logger_p.info("Running with the non-optimized version")
            pass

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

    def fuse_qkv_projections(self):
        try:
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()
            logger_p.info("QKV projection fusion")
        except Exception as e:
            logger_p.error(f"X Error merging QKV projections: {e}")
            pass

    def flash_attn(self):
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
                pass


class PipelineQwenImageEdit:
    def __init__(self, model_path: str | None, dist_inf: bool = False):
        self.pipeline: QwenImageEditPipeline | QwenImageEditPlusPipeline | None = None
        self.model_name = model_path
        self.pipelines = {}
        self.dist_inf = dist_inf

    def start(self):
        if torch.cuda.is_available():
            if self.model_name in [ImageModel.QWEN_IMAGE_EDIT_BASE]:
                self.pipeline = QwenImageEditPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16
                ).to("cuda")
            elif self.model_name in [ImageModel.QWEN_IMAGE_EDIT_2511, ImageModel.QWEN_IMAGE_EDIT_2509]:
                self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16
                ).to("cuda")
            else:
                raise ValueError("Unsupported model")
            self.optimization()
        else:
            raise ValueError("CUDA not available")

    def optimization(self):
        try:
            try:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.max_autotune_gemm = True
                torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
                torch._inductor.config.triton.cudagraphs = False
            except Exception as e:
                logger_p.error(f"X torch_opt failed: {str(e)}")
                pass
            self.flash_attn()
            self.fuse_qkv_projections()
            self.optimize_memory_format()
        except Exception as e:
            logger_p.error(f"X The optimizations could not be applied: {e}")
            logger_p.info("Running with the non-optimized version")
            pass

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

    def fuse_qkv_projections(self):
        try:
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()
            logger_p.info("QKV projection fusion")
        except Exception as e:
            logger_p.error(f"X Error merging QKV projections: {e}")
            pass

    def flash_attn(self):
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
                pass

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

class PipelineZImage:
    def __init__(self, model_path: str):
        self.model_name = model_path
        self.pipeline: ZImagePipeline | None = None

    def start(self):
        if torch.cuda.is_available():
            try:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.max_autotune_gemm = True
                torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
                torch._inductor.config.triton.cudagraphs = False
            except Exception as e:
                logger_p.error(f"X torch_opt failed: {str(e)}")
                pass
            self.pipeline = ZImagePipeline.from_pretrained(self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda")
            self.optimization()

    def optimization(self):
        self.optimize_memory_format()
        #self.flash_attn()

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

    def flash_attn(self):
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
                pass

class AutoPipelineDiffusers:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.pipeline: AutoPipelineForText2Image | None = None
        self.model_name = model_path
        self.dist_inf = dist_inf
        self.pipelines = {}
        

    def start(self):
        if torch.cuda.is_available():
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.model_name, device_map="cuda")
            self.optimization()

    def optimization(self):
        try:
            try:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True
                torch._inductor.config.max_autotune_gemm = True
                torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
                torch._inductor.config.triton.cudagraphs = False
            except Exception as e:
                logger_p.error(f"X torch_opt failed: {str(e)}")
                pass
            self.optimize_attention_sdpa()
            self.optimize_memory_format()
            self.fuse_qkv_projections()
        except Exception as e:
            logger_p.error(f"X The optimizations could not be applied: {e}")
            logger_p.info("Running with the non-optimized version")
            pass

    def optimize_attention_sdpa(self):
        try:
            logger_p.info("SDPA (Scaled Dot Product Attention)")
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
        except Exception as e:
            logger_p.error(f"X Error enabling SDPA: {e}")
            pass

    def optimize_memory_format(self): 
        try:
            logger_p.info("channels_last memory format")
            if hasattr(self.pipeline, 'unet'):
                self.pipeline.unet.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer.to(memory_format=torch.channels_last)
        except Exception as e:
            logger_p.error(f"X Error optimizing memory format: {e}")
            pass

    def fuse_qkv_projections(self):        
        try:
            self.pipeline.fuse_qkv_projections()
            logger_p.info("QKV projection fusion")
        except AttributeError:
            logger_p.warning("fuse_qkv_projections not available for this model")
            pass
        except Exception as e:
            logger_p.error(f"X Error merging QKV projections: {e}")
            pass

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