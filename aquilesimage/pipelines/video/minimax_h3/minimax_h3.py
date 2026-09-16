from typing import Literal
try:
    from diffusers import ModularPipeline
    from diffusers.utils.export_utils import encode_video
except ImportError as e:
    print(f"Error importing diffusers MiniMax-H3 components: {e}")
    ModularPipeline = None
    encode_video = None
import torch
import gc
import logging

logger_p = logging.getLogger("Aquiles-Image-Pipelines")

REPO_ID = "MiniMaxAI/MiniMax-H3"

# Layers kept in bf16 when the conditioner is quantized to int8 (docs recipe).
TEXT_ENCODER_SKIP_MODULES = [
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
]


class MiniMax_H3_Pipeline:
    """Video + audio pipeline based on diffusers ModularPipeline.

    Single pipeline for text-to-video (``t2va``) and first-frame
    image-to-video (``fl2va`` with ``image`` only): no ``image`` is
    T2V, one ``image`` is I2V anchored at the start. ``last_image``
    is intentionally unsupported.

    No ``from_pretrained(workflow=...)`` filter is used so the
    pipeline picks the workflow per call from the inputs. Loading
    with ``workflow="t2va"`` fetches only the ``transformer/``
    partition plus the shared components, which also serves
    ``fl2va``; ``transformer_ref/`` is never touched.
    """

    ATTENTION_BACKEND_PRIORITY: tuple[str, ...] = ("sage_hub",)

    def __init__(self, model_name: Literal["minimax-h3"] = "minimax-h3"):
        if model_name != "minimax-h3":
            raise ValueError("Model not available")
        self.pipeline: ModularPipeline | None = None
        self.model_name = model_name
        self.repo_id = REPO_ID
        self.frame_rate = 24.0

    def _load_quantized_text_encoder(self):
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import TorchAoConfig as TransformersTorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig

        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.repo_id,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=TEXT_ENCODER_SKIP_MODULES,
            ),
            low_cpu_mem_usage=False,
        )

    def start(self):
        if ModularPipeline is None:
            raise ImportError("diffusers MiniMax-H3 support is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MiniMax-H3")

        try:
            text_encoder = self._load_quantized_text_encoder()
            self.pipeline = ModularPipeline.from_pretrained(self.repo_id)
            self.pipeline.update_components(text_encoder=text_encoder)
        except Exception as e:
            print(f"Quantized text encoder failed, falling back to full bf16: {e}")
            self.pipeline = ModularPipeline.from_pretrained(self.repo_id)

        self.pipeline.load_components(workflow="t2va", dtype=torch.bfloat16)
        self.pipeline.to("cuda")

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
            try:
                transformer.set_attention_backend(backend)
                logger_p.info(f"Attention backend enabled: {backend}")
                return
            except Exception as e:
                logger_p.debug(f"Failed to set attention backend {backend}: {str(e)}")

        logger_p.warning("No optimized attention available, using default SDPA")

    # shared helpers

    def _resolve_num_frames(self, seconds=None) -> int:
        try:
            target = int(seconds) if seconds is not None else 5
        except (TypeError, ValueError):
            target = 5
        target = max(5, min(15, target))
        # Snap up to the next 17 * n + 5 the video VAE can decode.
        frames = target * int(self.frame_rate)
        n = (frames - 5 + 16) // 17
        return max(22, 17 * n + 5)

    def _build_generator(self, seed: int) -> torch.Generator:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=device).manual_seed(int(seed))

    def _check_started(self):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

    def _save_video(self, video, audio, sample_rate, save_result_path: str):
        encode_video(
            video[0],
            fps=self.frame_rate,
            audio=audio[0].float().cpu(),
            audio_sample_rate=int(sample_rate),
            output_path=save_result_path,
        )

    def _release_memory(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
        gc.collect()

    # generate dispatcher

    def generate(self, seed: int, prompt: str, save_result_path: str, negative_prompt: str, image=None, seconds=None):
        try:
            import os
            output_dir = os.path.dirname(save_result_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            self._check_started()

            num_frames = self._resolve_num_frames(seconds)
            generator = self._build_generator(seed)
            # Guidance-distilled weights: no guider, negative_prompt unused.
            call_kwargs = dict(
                prompt=prompt,
                num_frames=num_frames,
                generator=generator,
                output=["videos", "audio", "sampling_rate"],
            )
            if image is not None:
                call_kwargs["image"] = image

            with torch.inference_mode():
                results = self.pipeline(**call_kwargs)
                self._save_video(
                    results["videos"],
                    results["audio"],
                    results["sampling_rate"],
                    save_result_path,
                )

            print(f"Saved video in... {save_result_path}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

            raise

        finally:
            self._release_memory()
