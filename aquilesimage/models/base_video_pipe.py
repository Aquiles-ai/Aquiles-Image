import gc
import logging
from abc import abstractmethod

import torch

try:
    from diffusers.utils.export_utils import encode_video
except ImportError as e:
    print(f"Error importing diffusers video export utils: {e}")
    encode_video = None

from aquilesimage.models.base_pipe import BasePipeline

logger_p = logging.getLogger("Aquiles-Image-Pipelines")


class BaseVideoPipeline(BasePipeline):
    """Shared base for the diffusers-native video pipelines (LTX, MiniMax-H3).

    Contract with the video server: ``start()`` loads, ``generate()``
    writes an mp4 to ``save_result_path``. Everything equal in both
    pipelines lives here; each subclass overrides only its specifics
    (repos, workflows, schedulers, frame math) plus any hook below,
    e.g. ``ATTENTION_BACKEND_PRIORITY``.
    """

    FRAME_RATE: float = 24.0

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        seed: int,
        prompt: str,
        save_result_path: str,
        negative_prompt: str,
        image=None,
        seconds=None,
    ):
        """Runs inference and saves the resulting video + audio to disk."""

    def optimization(self):
        self.enable_flash_attn()

    def _build_generator(self, seed: int) -> torch.Generator:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=device).manual_seed(int(seed))

    def _check_started(self):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

    def _save_video(self, video, audio, sample_rate, save_result_path: str):
        if encode_video is None:
            raise ImportError("diffusers video export utils are not available")
        encode_video(
            video[0],
            fps=self.FRAME_RATE,
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
