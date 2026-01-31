from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
try:
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
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
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
try:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
except ImportError as e:
    print("Error import QwenImageEditPlusPipeline")
    pass
try:
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
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
from transformers import Qwen3ForCausalLM
try:
    from diffusers.pipelines.glm_image.pipeline_glm_image import GlmImagePipeline
    from transformers import T5EncoderModel, ByT5Tokenizer, GlmImageProcessor, GlmImageForConditionalGeneration
    from diffusers.models.transformers.transformer_glm_image import GlmImageTransformer2DModel
except ImportError as e:
    print("Error import GlmImagePipeline")
    pass
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

"""
Maybe this will mutate with the changes implemented in diffusers
"""

class PipelineSD3:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline | None = None
        self.device: str | None = None
        self.dist_inf = dist_inf
        self.pipelines = {}

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
                print(f"X xformers not available: {e}")

            try:
                self.enable_flash_attn()
            except Exception as e:
                print(f"X flash_attn not available: {e}")
                pass

        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

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

class PipelineFlux:
    def __init__(self, model_path: str | None = None, low_vram: bool = False, compile_flag: bool = False, dist_inf: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram
        self.compile_flag = compile_flag
        self.dist_inf = dist_inf
        self.pipelines = {}

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

            self.optimization()

                
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger_p.info("Loading MPS for Mac M Series")
            self.device = "mps"
            
        else:
            raise Exception("No CUDA or MPS device available")

    def optimization(self):
        try:
            logger_p.info("Starting optimization process...")
            
            config = torch._inductor.config
            config.conv_1x1_as_mm = True
            config.coordinate_descent_check_all_directions = True
            config.coordinate_descent_tuning = True
            config.disable_progress = False
            config.epilogue_fusion = False
            config.shape_padding = True

            logger_p.info("Fusing QKV projections...")
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()

            logger_p.info("Converting to channels_last memory format...")
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)

            logger_p.info("FlashAttention")
            self.enable_flash_attn()

            if self.compile_flag:
                logger_p.info("Compiling transformer and VAE...")
                self.pipeline.transformer = torch.compile(
                    self.pipeline.transformer,
                    mode="max-autotune-no-cudagraphs", 
                    dynamic=True
                )

                self.pipeline.vae.decode = torch.compile(
                    self.pipeline.vae.decode, 
                    mode="max-autotune-no-cudagraphs", 
                    dynamic=True
                )

                logger_p.info("Triggering torch.compile with dummy inference...")
                _ = self.pipeline(
                    "dummy prompt",
                    height=1024,
                    width=1024,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                ).images[0]
            
                logger_p.info("Compilation trigger completed")

                self._warmup()
            
            logger_p.info("All optimizations completed successfully")
            
        except Exception as e:
            logger_p.error(f"X Error in optimization with Flux: {e}")
            raise

    def _warmup(self):
        try:
            logger_p.info("Starting warmup process...")
            warmup_prompt = "a simple test image"
            for i in range(3):
                _ = self.pipeline(
                    prompt=warmup_prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=4,
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
            pass

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


class PipelineFluxKontext:
    def __init__(self, model_path: str | None = None, low_vram: bool = False, dist_inf: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxKontextPipeline | None = None
        self.device: str | None = None
        self.low_vram = low_vram
        self.dist_inf = dist_inf
        self.pipelines = {}

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

            self.optimization()
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

    def optimization(self):
        try:
            logger_p.info("Starting optimization process...")
            
            config = torch._inductor.config
            config.conv_1x1_as_mm = True
            config.coordinate_descent_check_all_directions = True
            config.coordinate_descent_tuning = True
            config.disable_progress = False
            config.epilogue_fusion = False
            config.shape_padding = True

            logger_p.info("Fusing QKV projections...")
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()

            logger_p.info("Converting to channels_last memory format...")
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)

            logger_p.info("FlashAttention")
            self.enable_flash_attn()
            
            logger_p.info("All optimizations completed successfully")
            
        except Exception as e:
            logger_p.error(f"X Error in optimization with FluxKontext: {e}")
            raise

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

class PipelineFlux2Klein:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.model_name = model_path
        if dist_inf:
            raise ValueError("Distributed inference is not supported for these models")
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