from typing import Literal
try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2 import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
except ImportError as e:
    print(f"Error importing diffusers LTX-2 components: {e}")
    FlowMatchEulerDiscreteScheduler = None
    LTX2ConditionPipeline = None
    LTX2LatentUpsamplePipeline = None
    LTX2LatentUpsamplerModel = None
    LTX2VideoCondition = None
    DEFAULT_NEGATIVE_PROMPT = "No deformities"
    DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
    STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875]
import torch
import logging
from aquilesimage.models import BaseVideoPipeline

logger_p = logging.getLogger("Aquiles-Image-Pipelines")

REPO_MAP = {
    "ltx-2": "Lightricks/LTX-2",
    "ltx-2.3": "diffusers/LTX-2.3-Diffusers",
    "ltx-2.5": "Lightricks/LTX-2.5-Diffusers",
}

# Stage 2 distilled LoRA at the root of each pipeline repo.
# Unused by the ltx-2.5 distilled flow (transformer is already distilled).
STAGE_2_LORA_MAP = {
    "ltx-2": "ltx-2-19b-distilled-lora-384.safetensors",
    "ltx-2.3": "ltx-2.3-22b-distilled-lora-384.safetensors",
    "ltx-2.5": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
}

# Quantized text encoder pinned for loading: QAT checkpoint + BnB 4-bit.
# Falls back to the full bf16 encoder from the pipeline repo on failure.
TEXT_ENCODER_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


class LTX_2_Pipeline(BaseVideoPipeline):
    """Video pipeline based on diffusers.

    Single pipeline for text-to-video and image-to-video via
    ``LTX2ConditionPipeline``: empty ``conditions`` is T2V, one
    ``LTX2VideoCondition`` at index 0 is I2V.

    ltx-2: two-stage generation (stage 1 base DiT at 768x512,
    latent upsample x2, stage 2 refine at 1536x1024 with
    distilled LoRA + distilled sigmas).
    ltx-2.3: single-stage only for now (diffusers/LTX-2.3-Diffusers
    ships neither latent_upsampler/ nor the stage-2 LoRA).
    ltx-2.5: two-stage distilled (default transformer is already
    distilled, no LoRA; stage 1 half res with distilled sigmas,
    upsample x2, stage 2 refine at full res).
    """

    # Flash backends discarded: LTX-2 connectors pass `attn_mask` and
    # flash-attn 2 raises `ValueError: attn_mask is not supported`.
    ATTENTION_BACKEND_PRIORITY: tuple[str, ...] = ("sage_hub",)

    def __init__(self, model_name: Literal["ltx-2", "ltx-2.3", "ltx-2.5"] = "ltx-2"):
        if model_name not in REPO_MAP:
            raise ValueError("Model not available")
        super().__init__(model_name)
        self.pipeline: LTX2ConditionPipeline | None = None
        self.repo_id = REPO_MAP[model_name]
        self.stage_2_lora = STAGE_2_LORA_MAP[model_name]
        self.use_two_stage = (model_name in ("ltx-2", "ltx-2.5"))
        self.is_distilled_flow = (model_name == "ltx-2.5")
        self.use_lora_stage2 = (model_name == "ltx-2")
        self._base_scheduler = None
        self._stage_2_scheduler = None
        self.seconds_map = {
            "4": 121,
            "8": 193,
            "12": 289,
        }
        # ltx-2 final output is 1536x1024 (stage 1 at 768x512, x2 refine).
        # ltx-2.3 single-stage output is 768x512.
        # ltx-2.5 distilled target is 1536x1024 (stage 1 at 768x512, x2 refine).
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

    # start flows

    def start(self):
        if LTX2ConditionPipeline is None:
            raise ImportError("diffusers LTX-2 support is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LTX-2")

        self._load_base_pipeline()
        if not self.use_two_stage:
            logger_p.info(f"{self.model_name}: single-stage mode, skipping stage-2 LoRA/upsampler")
            return
        self._load_stage2_assets()

    def _load_base_pipeline(self):
        if self.is_distilled_flow:
            # ltx-2.5 ships Gemma 4 + distilled scheduler config, load as is.
            self.pipeline = LTX2ConditionPipeline.from_pretrained(
                self.repo_id, dtype=torch.bfloat16
            ).to("cuda")
        else:
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

        # Schedulers: base for stage 1 / single-stage, distilled config for stage 2.
        self._base_scheduler = self.pipeline.scheduler
        self._stage_2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

    def _load_stage2_assets(self):
        if self.is_distilled_flow:
            # Distilled transformer needs no LoRA, only the upsampler.
            latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                self.repo_id, subfolder="latent_upsampler", dtype=torch.bfloat16
            )
            self.upsample_pipe = LTX2LatentUpsamplePipeline(
                vae=self.pipeline.vae, latent_upsampler=latent_upsampler
            ).to("cuda")
            return
        # Stage 2 distilled LoRA (same repo, no extra big download).
        self.pipeline.load_lora_weights(
            self.repo_id,
            adapter_name="stage_2_distilled",
            weight_name=self.stage_2_lora,
        )
        self.pipeline.disable_lora()

        # Latent upsampler x2 (subfolder of the same repo).
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.repo_id, subfolder="latent_upsampler", dtype=torch.bfloat16
        )
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipeline.vae, latent_upsampler=latent_upsampler
        ).to("cuda")

    # shared helpers

    def _resolve_num_frames(self, seconds=None) -> int:
        if seconds is None:
            return 121
        return self.seconds_map.get(str(seconds), 121)

    def _resolve_negative(self, negative_prompt: str) -> str:
        # The server sends a weak hardcoded negative ("No deformities");
        # prefer the docs' full negative prompt for LTX in that case.
        if not negative_prompt or negative_prompt.strip().lower() == "no deformities":
            return DEFAULT_NEGATIVE_PROMPT
        return negative_prompt

    def _build_conditions(self, image):
        if image is not None:
            return [LTX2VideoCondition(frames=image, index=0, strength=1.0)]
        return []

    def _check_started(self):
        super()._check_started()
        if self.use_two_stage and self.upsample_pipe is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

    def _restore_state(self):
        if self.pipeline is not None:
            try:
                if self.use_lora_stage2:
                    self.pipeline.disable_lora()
                if self._base_scheduler is not None:
                    self.pipeline.scheduler = self._base_scheduler
            except Exception:
                pass

    # ltx-2.3 single-stage

    def _run_single_stage(self, conditions, prompt, negative, num_frames, generator):
        return self.pipeline(
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
            output_type="np",
            return_dict=False,
        )

    # ltx-2 two-stage

    def _run_stage1_latent(self, conditions, prompt, negative, num_frames, generator):
        self.pipeline.disable_lora()
        self.pipeline.scheduler = self._base_scheduler
        return self.pipeline(
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

    def _run_upsample(self, video_latent):
        return self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

    def _run_stage2_refine(self, conditions, prompt, negative, upscaled_latent,
                           audio_latent, num_frames, generator):
        # upscaled_latent is 1536x1024, so width/height must be x2
        # or prepare_latents() computes mask_shape from 768x512
        # (6144 tokens) and rejects the upscaled latents (24576 tokens).
        self.pipeline.enable_lora()
        self.pipeline.set_adapters("stage_2_distilled", 1.0)
        self.pipeline.scheduler = self._stage_2_scheduler
        return self.pipeline(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative,
            latents=upscaled_latent,
            audio_latents=audio_latent,
            width=self.width * 2,
            height=self.height * 2,
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

    # ltx-2.5 distilled two-stage (default transformer, no LoRA)

    def _run_stage1_distilled_latent(self, conditions, prompt, negative, num_frames, generator):
        return self.pipeline(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative,
            width=self.width,
            height=self.height,
            num_frames=num_frames,
            frame_rate=self.frame_rate,
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )

    def _run_upsample_distilled(self, video_latent):
        return self.upsample_pipe(
            latents=video_latent,
            latents_normalized=False,
            output_type="latent",
            return_dict=False,
        )[0]

    def _run_stage2_distilled_refine(self, conditions, prompt, negative, upscaled_latent,
                                     audio_latent, num_frames, generator):
        return self.pipeline(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative,
            latents=upscaled_latent,
            audio_latents=audio_latent,
            width=self.width * 2,
            height=self.height * 2,
            num_frames=num_frames,
            frame_rate=self.frame_rate,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
        )

    def _generate_distilled_two_stage(self, conditions, prompt, negative, num_frames, generator):
        video_latent, audio_latent = self._run_stage1_distilled_latent(
            conditions, prompt, negative, num_frames, generator
        )
        upscaled_latent = self._run_upsample_distilled(video_latent)
        return self._run_stage2_distilled_refine(
            conditions, prompt, negative, upscaled_latent,
            audio_latent, num_frames, generator,
        )

    def _generate_two_stage(self, conditions, prompt, negative, num_frames, generator):
        if self.is_distilled_flow:
            return self._generate_distilled_two_stage(
                conditions, prompt, negative, num_frames, generator
            )
        video_latent, audio_latent = self._run_stage1_latent(
            conditions, prompt, negative, num_frames, generator
        )
        upscaled_latent = self._run_upsample(video_latent)
        return self._run_stage2_refine(
            conditions, prompt, negative, upscaled_latent,
            audio_latent, num_frames, generator,
        )

    # generate dispatcher

    def generate(self, seed: int, prompt: str, save_result_path: str, negative_prompt: str, image=None, seconds=None):
        try:
            import os
            output_dir = os.path.dirname(save_result_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            self._check_started()

            num_frames = self._resolve_num_frames(seconds)
            negative = self._resolve_negative(negative_prompt)
            conditions = self._build_conditions(image)
            generator = self._build_generator(seed)

            with torch.inference_mode():
                if not self.use_two_stage:
                    video, audio = self._run_single_stage(
                        conditions, prompt, negative, num_frames, generator
                    )
                else:
                    video, audio = self._generate_two_stage(
                        conditions, prompt, negative, num_frames, generator
                    )

                self._save_video(
                    video, audio,
                    self.pipeline.vocoder.config.output_sampling_rate,
                    save_result_path,
                )

            print(f"Saved video in... {save_result_path}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

            raise

        finally:
            self._restore_state()
            self._release_memory()
