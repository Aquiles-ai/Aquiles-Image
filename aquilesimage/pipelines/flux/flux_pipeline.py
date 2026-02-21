import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from aquilesimage.utils import setup_colored_logger
import os
import logging

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

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