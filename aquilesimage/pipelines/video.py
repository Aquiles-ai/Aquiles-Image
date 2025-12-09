import torch
from lightx2v import LightX2VPipeline
from aquilesimage.utils import get_path_file_video_model, file_exists, download_base_wan_2_2

class Wan2_2_Pipeline:
    def __init__(self, h: int = 720, w: int = 1280, frames: int = 81):
        self.pipeline: LightX2VPipeline | None = None
        self.h = h
        self.w = w
        self.frames = frames
        self.verify_model()

    def start(self):
        if torch.cuda.is_available():
            self.pipeline = LightX2VPipeline(
                model_path=get_path_file_video_model("wan2.2"),
                model_cls="wan2.2_moe",
                task="t2v",
            )

            self.pipeline.create_generator(
                attn_mode="flash_attn2",
                infer_steps=40,
                num_frames=self.frames,
                height=self.h,
                width=self.w,
                guidance_scale=[3.5, 3.5],
                sample_shift=5.0,
            )
        else:
            raise Exception("No CUDA device available")

    def verify_model(self):
        model_path = get_path_file_video_model("wan2.2")

        if(file_exists(model_path)):
            pass
        else:
            download_base_wan_2_2()

class ModelVideoPipelineInit:
    def __init__(self, model: str):
        self.model = model
        self.pipeline = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        if self.model == 'wan2.2':
            self.pipeline = Wan2_2_Pipeline()

        return self.pipeline