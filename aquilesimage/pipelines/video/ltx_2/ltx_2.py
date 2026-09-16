from typing import Literal
try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2 import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
    from diffusers.utils import encode_video
except ImportError as e:
    print(f"Error importing diffusers LTX-2 components: {e}")
    FlowMatchEulerDiscreteScheduler = None
    LTX2ConditionPipeline = None
    LTX2LatentUpsamplePipeline = None
    LTX2LatentUpsamplerModel = None
    LTX2VideoCondition = None
    DEFAULT_NEGATIVE_PROMPT = "No deformities"
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875]
    encode_video = None
import torch
import gc
import logging

logger_p = logging.getLogger("Aquiles-Image-Pipelines")

REPO_MAP = {
    "ltx-2": "Lightricks/LTX-2",
    "ltx-2.3": "diffusers/LTX-2.3-Diffusers",
}

# Stage 2 distilled LoRA shipped at the root of each pipeline repo.
STAGE_2_LORA_MAP = {
    "ltx-2": "ltx-2-19b-distilled-lora-384.safetensors",
    "ltx-2.3": "ltx-2.3-22b-distilled-lora-384.safetensors",
}

# Quantized text encoder pinned for loading: QAT checkpoint + BnB 4-bit.
# Falls back to the full bf16 encoder from the pipeline repo on failure.
TEXT_ENCODER_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


class LTX_2_Pipeline:
    """Video pipeline based on diffusers.

    Single pipeline for text-to-video and image-to-video via
    ``LTX2ConditionPipeline``: empty ``conditions`` is T2V, one
    ``LTX2VideoCondition`` at index 0 is I2V.

    Two-stage generation: stage 1 at half resolution (base DiT),
    latent upsample x2, stage 2 refine at full resolution
    (distilled LoRA + distilled sigmas).
    """

    # Flash backends discarded: LTX-2 connectors pass `attn_mask` and
    # flash-attn 2 raises `ValueError: attn_mask is not supported`.
    ATTENTION_BACKEND_PRIORITY: tuple[str, ...] = ("sage_hub",)

    def __init__(self, model_name: Literal["ltx-2", "ltx-2.3"] = "ltx-2"):
        if model_name not in REPO_MAP:
            raise ValueError("Model not available")
        self.pipeline: LTX2ConditionPipeline | None = None
        self.upsample_pipe: LTX2LatentUpsamplePipeline | None = None
        self.model_name = model_name
        self.repo_id = REPO_MAP[model_name]
        self.stage_2_lora = STAGE_2_LORA_MAP[model_name]
        self._base_scheduler = None
        self._stage_2_scheduler = None
        self.seconds_map = {
            "4": 121,
            "8": 193,
            "12": 289,
        }
        # Full-resolution output; stage 1 runs at half, stage 2 refines full.
        self.width = 768
        self.height = 512
        self.frame_rate = 24.0
        self.num_inference_steps = 30

    def _load_quantized_text_encoder(self):
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        return Gemma3ForConditionalGeneration.from_pretrained(
            TEXT_ENCODER_REPO,
            quantization_config=quant,
            dtype=torch.bfloat16,
        )

    def start(self):
        if LTX2ConditionPipeline is None:
            raise ImportError("diffusers LTX-2 support is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LTX-2")

        try:
            text_encoder = self._load_quantized_text_encoder()
            self.pipeline = LTX2ConditionPipeline.from_pretrained(
                self.repo_id, text_encoder=text_encoder, dtype=torch.bfloat16
            ).to("cuda")
        except Exception as e:
            print(f"Quantized text encoder failed, falling back to full bf16: {e}")
            self.pipeline = LTX2ConditionPipeline.from_pretrained(
                self.repo_id, dtype=torch.bfloat16
            ).to("cuda")

        if hasattr(self.pipeline, "vae") and hasattr(self.pipeline.vae, "enable_tiling"):
            self.pipeline.vae.enable_tiling()

        # Stage 2 distilled LoRA (same repo, no extra big download).
        self.pipeline.load_lora_weights(
            self.repo_id,
            adapter_name="stage_2_distilled",
            weight_name=self.stage_2_lora,
        )
        self.pipeline.disable_lora()

        # Schedulers: base for stage 1, distilled config for stage 2.
        self._base_scheduler = self.pipeline.scheduler
        self._stage_2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

        # Latent upsampler x2 (subfolder of the same repo).
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.repo_id, subfolder="latent_upsampler", dtype=torch.bfloat16
        )
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipeline.vae, latent_upsampler=latent_upsampler
        ).to("cuda")

    def enable_flash_attn(self):
        if self.pipeline is None:
            logger_p.warning("No pipeline loaded, skipping flash attention")
            return

        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is None:
            logger_p.warning("No transformer component found for flash attention")
            return

        if not hasattr(transformer, "set_attention_backend"):
            logger_p.warning(
                "set_attention_backend not available for this model, skipping flash attention"
            )
            return

        for backend in self.ATTENTION_BACKEND_PRIORITY:
            if not self._attention_backend_ready(backend):
                logger_p.debug(f"Attention backend {backend} not available")
                continue
            try:
                transformer.set_attention_backend(backend)
                logger_p.info(f"Attention backend enabled: {backend}")
                return
            except Exception as e:
                logger_p.debug(f"Failed to set attention backend {backend}: {str(e)}")

        logger_p.warning("No optimized attention available, using default SDPA")

    def _attention_backend_ready(self, backend: str) -> bool:
        try:
            from diffusers.models.attention_dispatch import (
                AttentionBackendName,
                _HUB_KERNELS_REGISTRY,
                _check_attention_backend_requirements,
            )
            name = AttentionBackendName(backend)
            _check_attention_backend_requirements(name)
            if name in _HUB_KERNELS_REGISTRY:
                config = _HUB_KERNELS_REGISTRY[name]
                return self._hub_kernel_ready(config.repo_id, config.version)
            return True
        except Exception as e:
            logger_p.debug(f"Attention backend {backend} not usable: {str(e)}")
            return False

    def _hub_kernel_ready(self, repo_id: str, version: int | None) -> bool:
        try:
            from kernels import get_kernel, has_kernel
        except Exception as e:
            logger_p.debug(f"kernels package not usable: {str(e)}")
            return False
        try:
            if has_kernel(repo_id, version=version):
                return True
            get_kernel(repo_id, version=version)
            return True
        except Exception as e:
            logger_p.debug(f"Hub kernel {repo_id} not usable: {str(e)}")
            return False

    def _resolve_num_frames(self, seconds=None) -> int:
        if seconds is None:
            return 121
        return self.seconds_map.get(str(seconds), 121)

    def generate(self, seed: int, prompt: str, save_result_path: str, negative_prompt: str, image=None, seconds=None):
        try:
            import os
            output_dir = os.path.dirname(save_result_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            if self.pipeline is None or self.upsample_pipe is None:
                raise RuntimeError("Pipeline not started. Call start() first.")

            num_frames = self._resolve_num_frames(seconds)
            # The server sends a weak hardcoded negative ("No deformities");
            # prefer the docs' full negative prompt for LTX in that case.
            if not negative_prompt or negative_prompt.strip().lower() == "no deformities":
                negative = DEFAULT_NEGATIVE_PROMPT
            else:
                negative = negative_prompt

            if image is not None:
                conditions = [LTX2VideoCondition(frames=image, index=0, strength=1.0)]
            else:
                conditions = []

            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

            with torch.inference_mode():
                # Stage 1: base DiT at full resolution, latent output.
                # The upsampler doubles it, so stage 2 refines at 1536x1024.
                self.pipeline.disable_lora()
                self.pipeline.scheduler = self._base_scheduler
                video_latent, audio_latent = self.pipeline(
                    conditions=conditions,
                    prompt=prompt,
                    negative_prompt=negative,
                    width=self.width,
                    height=self.height,
                    num_frames=num_frames,
                    frame_rate=self.frame_rate,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=3.0,
                    stg_scale=1.0,
                    modality_scale=3.0,
                    guidance_rescale=0.7,
                    audio_guidance_scale=7.0,
                    audio_stg_scale=1.0,
                    audio_modality_scale=3.0,
                    audio_guidance_rescale=0.7,
                    spatio_temporal_guidance_blocks=[28],
                    use_cross_timestep=True,
                    generator=generator,
                    output_type="latent",
                    return_dict=False,
                )

                # Upsample latents x2.
                upscaled_latent = self.upsample_pipe(
                    latents=video_latent,
                    output_type="latent",
                    return_dict=False,
                )[0]

                # Stage 2: distilled LoRA + distilled sigmas at full resolution.
                self.pipeline.enable_lora()
                self.pipeline.set_adapters("stage_2_distilled", 1.0)
                self.pipeline.scheduler = self._stage_2_scheduler
                video, audio = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative,
                    latents=upscaled_latent,
                    audio_latents=audio_latent,
                    width=self.width,
                    height=self.height,
                    num_frames=num_frames,
                    frame_rate=self.frame_rate,
                    num_inference_steps=3,
                    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                    guidance_scale=1.0,
                    audio_guidance_scale=1.0,
                    generator=generator,
                    output_type="np",
                    return_dict=False,
                )

                audio_tensor = audio[0].float().cpu()
                audio_sample_rate = self.pipeline.vocoder.config.output_sampling_rate

                encode_video(
                    video[0],
                    fps=self.frame_rate,
                    audio=audio_tensor,
                    audio_sample_rate=audio_sample_rate,
                    output_path=save_result_path,
                )

            print(f"Saved video in... {save_result_path}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

            raise

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.disable_lora()
                    if self._base_scheduler is not None:
                        self.pipeline.scheduler = self._base_scheduler
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            gc.collect()
