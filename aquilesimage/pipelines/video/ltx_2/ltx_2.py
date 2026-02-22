from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_ltx_2
from typing import Literal
try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.components.guiders import MultiModalGuiderParams
except ImportError as e:
    print("Error importing components for LTX-2")
    pass
import torch
import gc

class LTX_2_Pipeline:
    def __init__(self, model_name: Literal["ltx-2"] = "ltx-2"):
        self.pipeline: TI2VidTwoStagesPipeline | None = None
        self.model_name = model_name
        self.verify_model()

    def start(self):
        data_dir = get_path_file_video_model(self.model_name)

        with torch.no_grad():
            self.pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=f"{data_dir}/ltx-2-19b-dev.safetensors",
                gemma_root=f"{data_dir}/gemma", 
                loras=[], 
                distilled_lora=[
                    LoraPathStrengthAndSDOps(
                        path=f"{data_dir}/ltx-2-19b-distilled-lora-384.safetensors", 
                        strength=0.6, 
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP
                    )
                ], 
                spatial_upsampler_path=f"{data_dir}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
            )

    def generate(self, seed: int, prompt: str, save_result_path: str, negative_prompt: str):
        try:
            import os
            output_dir = os.path.dirname(save_result_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with torch.no_grad():
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(300, tiling_config)

                video, audio = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=1088,
                    width=1920,
                    num_frames=300,
                    frame_rate=25.0,
                    num_inference_steps=40,
                    images=[],
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=3.0,
                        stg_scale=1.0,
                        rescale_scale=0.7,
                        modality_scale=3.0,
                        skip_step=0,
                        stg_blocks=[29],
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=7.0,
                        stg_scale=1.0,
                        rescale_scale=0.7,
                        modality_scale=3.0,
                        skip_step=0,
                        stg_blocks=[29],
                    ),
                    enhance_prompt=False,
                    tiling_config=tiling_config
                )

                encode_video(
                    video=video,
                    fps=25.0,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=save_result_path,
                    video_chunks_number=video_chunks_number,
                )

            print(f"Saved video in... {save_result_path}")

        except Exception as e:
            print(f"X Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            gc.collect()

    def verify_model(self):
        model_path = get_path_file_video_model(self.model_name)

        if (file_exists(f"{model_path}/gemma/model-00004-of-00005.safetensors") and 
            file_exists(f"{model_path}/ltx-2-19b-dev.safetensors") and 
            file_exists(f"{model_path}/ltx-2-spatial-upscaler-x2-1.0.safetensors") and 
            file_exists(f"{model_path}/ltx-2-19b-distilled-lora-384.safetensors")):
            pass
        else:
            download_ltx_2()