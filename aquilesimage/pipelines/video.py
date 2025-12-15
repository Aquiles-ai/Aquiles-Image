import torch
from lightx2v import LightX2VPipeline
from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_base_wan_2_2, download_wan_2_2_turbo

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

            self.pipeline.text_len = 512

            self.pipeline.enable_cfg = True

            self.pipeline.create_generator(
                attn_mode="flash_attn2",
                infer_steps=40,
                num_frames=self.frames,
                height=self.h,
                width=self.w,
                guidance_scale=[3.5, 3.5],
                sample_shift=12.0, 
            )
        else:
            raise Exception("No CUDA device available")

    def verify_model(self):
        model_path = get_path_file_video_model("wan2.2")

        if(file_exists(f"{model_path}/Wan2.1_VAE.pth")):
            pass
        else:
            download_base_wan_2_2()

class Wan2_2_Turbo_Pipeline:
    def __init__(self, h: int = 720, w: int = 1280, frames: int = 81):
        self.pipeline: LightX2VPipeline | None = None
        self.h = h
        self.w = w
        self.frames = frames
        self.verify_model()

    def start(self):
        if torch.cuda.is_available():
            self.pipeline = LightX2VPipeline(
                model_path=get_path_file_video_model("wan2.2-turbo"),
                model_cls="wan2.2_moe",
                task="t2v",
            )

            self.pipeline.text_len = 512

            self.pipeline.enable_cfg = False

            self.pipeline._class_name = "WanModel"

            self.pipeline.dim = 5120

            self.pipeline.eps = 1e-06

            self.pipeline.ffn_dim = 13824

            self.pipeline.freq_dim = 256

            self.pipeline.in_dim = 16

            self.pipeline.model_type = "t2v"

            self.pipeline.num_heads = 40

            self.pipeline.num_layers = 40

            self.pipeline.out_dim = 16

            self.pipeline.create_generator(
                attn_mode="flash_attn2",
                infer_steps=4,
                num_frames=self.frames,
                height=self.h,
                width=self.w,
                guidance_scale=[1.0, 1.0],
                sample_shift=5.0, 
            )
        else:
            raise Exception("No CUDA device available")

    def verify_model(self):
        model_path = get_path_file_video_model("wan2.2-turbo")

        if(file_exists(f"{model_path}/Wan2.1_VAE.pth")):
            pass
        else:
            download_wan_2_2_turbo()

class ModelVideoPipelineInit:
    def __init__(self, model: str):
        self.model = model
        self.pipeline = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        if self.model == 'wan2.2':
            self.pipeline = Wan2_2_Pipeline()

        elif self.model == 'wan2.2-turbo':
            self.pipeline = Wan2_2_Turbo_Pipeline()

        return self.pipeline