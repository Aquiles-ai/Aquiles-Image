import torch
try:
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
except ImportError as e:
    print("Error import ZImagePipeline")
    pass
from aquilesimage.utils import setup_colored_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import gc
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

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