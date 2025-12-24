import torch
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v import LightX2VPipeline
from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_base_wan_2_2, download_wan_2_2_turbo, download_hy, download_wan2_1
from typing import Literal

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

class HunyuanVideo_Pipeline:
    def __init__(self, model_name: Literal["hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p", "hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8", "hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"], frames: int = 81):
        self.pipeline: LightX2VPipeline | None = None
        self.frames = frames
        self.model_name = model_name
        self.verify_model()

    def start(self):
        if torch.cuda.is_available():
            if self.model_name in ["hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8"]:
                self.start_fp8(self.model_name)
            elif self.model_name in ["hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p"]:
                self.start_standard(self.model_name)
            elif self.model_name in ["hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"]:
                self.start_turbo(self.model_name)
        else:
            raise Exception("No CUDA device available")

    def verify_model(self):
        model_path = get_path_file_video_model(self.model_name)
    
        verification_files = {
            "hunyuanVideo-1.5-480p": "transformer/480p_t2v/diffusion_pytorch_model.safetensors",
            "hunyuanVideo-1.5-720p": "transformer/720p_t2v/diffusion_pytorch_model.safetensors",
            "hunyuanVideo-1.5-480p-fp8": "quantized/hy15_480p_t2v_fp8_e4m3_lightx2v.safetensors",
            "hunyuanVideo-1.5-720p-fp8": "quantized/hy15_720p_t2v_fp8_e4m3_lightx2v.safetensors",
            "hunyuanVideo-1.5-480p-turbo": "lora/hy1.5_t2v_480p_lightx2v_4step.safetensors",
            "hunyuanVideo-1.5-480p-turbo-fp8": "lora/hy1.5_t2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        }
    
        file_to_verify = verification_files.get(self.model_name)
    
        if file_to_verify is None:
            raise ValueError(f"Unrecognized model: {self.model_name}")

        full_path = f"{model_path}/{file_to_verify}"
    
        if file_exists(full_path):
            pass
        else:
            download_hy(self.model_name)

    def start_fp8(self, name: Literal["hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8"]):
        if name == "hunyuanVideo-1.5-480p-fp8":

            model_path = get_path_file_video_model("hunyuanVideo-1.5-480p-fp8")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="hunyuan_video_1.5",
                transformer_model_name="480p_t2v",
                task="t2v",
            )

            self.pipeline.use_image_encoder = False  

            self.pipeline.enable_quantize(  
                dit_quantized=True,  
                dit_quantized_ckpt=f"{model_path}/quantized/hy15_480p_t2v_fp8_e4m3_lightx2v.safetensors",  
                text_encoder_quantized=False, 
                quant_scheme="fp8-vllm",
                image_encoder_quantized=False,
            )

            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=50,  
                num_frames=121,  
                guidance_scale=6.0,  
                sample_shift=9.0, 
                fps=24,  
            )

            self.pipeline.runner.set_config({"aspect_ratio": "16:9"})


        elif name == "hunyuanVideo-1.5-720p-fp8":

            model_path = get_path_file_video_model("hunyuanVideo-1.5-720p-fp8")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="hunyuan_video_1.5",
                transformer_model_name="720p_t2v",
                task="t2v",
            )

            self.pipeline.use_image_encoder = False  

            self.pipeline.enable_quantize(  
                dit_quantized=True,  
                dit_quantized_ckpt=f"{model_path}/quantized/hy15_720p_t2v_fp8_e4m3_lightx2v.safetensors",  
                text_encoder_quantized=False, 
                quant_scheme="fp8-vllm",
                image_encoder_quantized=False,
            )

            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=50,  
                num_frames=121,  
                guidance_scale=6.0,  
                sample_shift=9.0, 
                height=720,
                width=1280,
                fps=24,  
            )

            self.pipeline.runner.set_config({"aspect_ratio": "16:9"})

    def start_standard(self, name: Literal["hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p"]):
        if name == "hunyuanVideo-1.5-480p":
            model_path = get_path_file_video_model("hunyuanVideo-1.5-480p")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="hunyuan_video_1.5",
                transformer_model_name="480p_t2v",
                task="t2v",
            )

            self.pipeline.use_image_encoder = False  

            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=50,  
                num_frames=121,  
                guidance_scale=6.0,  
                sample_shift=9.0,   
                fps=24,  
            )

            self.pipeline.runner.set_config({"aspect_ratio": "16:9"})

        elif name == "hunyuanVideo-1.5-720p":
            model_path = get_path_file_video_model("hunyuanVideo-1.5-720p")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="hunyuan_video_1.5",
                transformer_model_name="720p_t2v",
                task="t2v",
            )

            self.pipeline.use_image_encoder = False  

            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=50,  
                num_frames=121,  
                guidance_scale=6.0,  
                sample_shift=9.0,
                height=720,
                width=1280,
                fps=24,  
            )

            self.pipeline.runner.set_config({"aspect_ratio": "16:9"})



    def start_turbo(self, name: Literal["hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"]):

        if name == "hunyuanVideo-1.5-480p-turbo":
            model_path = get_path_file_video_model("hunyuanVideo-1.5-480p-turbo")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="hunyuan_video_1.5",
                transformer_model_name="480p_t2v",
                task="t2v",
                dit_original_ckpt=f"{model_path}/lora/hy1.5_t2v_480p_lightx2v_4step.safetensors",
            )

            self.pipeline.use_image_encoder = False

            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=4,  
                num_frames=81,  
                guidance_scale=1,  
                sample_shift=9.0,   
                fps=16,
                denoising_step_list=[1000, 750, 500, 250]
            )

            self.pipeline.runner.set_config({"aspect_ratio": "16:9"})


        elif name == "hunyuanVideo-1.5-480p-turbo-fp8":
            model_path = get_path_file_video_model("hunyuanVideo-1.5-480p-turbo-fp8")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="hunyuan_video_1.5",
                transformer_model_name="480p_t2v",
                task="t2v",
            )

            self.pipeline.use_image_encoder = False  

            self.pipeline.enable_quantize(
                quant_scheme="fp8-vllm",
                dit_quantized=True,
                dit_quantized_ckpt=f"{model_path}/lora/hy1.5_t2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors",
                image_encoder_quantized=False,
            )

            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=4,  
                num_frames=81,  
                guidance_scale=1,  
                sample_shift=9.0,  
                fps=16,
                denoising_step_list=[1000, 750, 500, 250]
            )

            self.pipeline.runner.set_config({"aspect_ratio": "16:9"})


class Wan2_1_Pipeline:
    def __init__(self, model_name: Literal["wan2.1", "wan2.1-3B", "wan2.1-turbo", "wan2.1-turbo-fp8"]):
        self.pipeline: LightX2VPipeline | None = None
        self.model_name = model_name
        self.verify_model()

    def verify_model(self):
        model_path = get_path_file_video_model(self.model_name)
    
        verification_files = {
            "wan2.1": "diffusion_pytorch_model-00001-of-00006.safetensors",
            "wan2.1-3B": "diffusion_pytorch_model.safetensors",
            "wan2.1-turbo": "lora/wan2.1_t2v_14b_lightx2v_4step.safetensors",
            "wan2.1-turbo-fp8": "lora/wan2.1_t2v_14b_scaled_fp8_e4m3_lightx2v_4step.safetensors",            
        }
    
        file_to_verify = verification_files.get(self.model_name)

        if file_to_verify is None:
            raise ValueError(f"Unrecognized model: {self.model_name}")

        full_path = f"{model_path}/{file_to_verify}"
    
        if file_exists(full_path):
            pass
        else:
            download_wan2_1(self.model_name)


    def start(self):
        if torch.cuda.is_available():
            if self.model_name in ["wan2.1", "wan2.1-3B"]:
                self.start_standard(self.model_name)
            elif self.model_name in ["wan2.1-turbo", "wan2.1-turbo-fp8"]:
                self.start_turbo(self.model_name)
        else:
            raise Exception("No CUDA device available")

    def start_standard(self, name: Literal["wan2.1", "wan2.1-3B"]):
        if name == "wan2.1":
            model_path = get_path_file_video_model("wan2.1")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="wan2.1",
                task="t2v",
            )
            
            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=40,  
                num_frames=81,  
                guidance_scale=5.0,  
                sample_shift=5.0,
                height=720,
                width=1280,
                fps=16,  
            )
        
        elif name == "wan2.1-3B":
            model_path = get_path_file_video_model("wan2.1-3B")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="wan2.1",
                task="t2v",
            )
            
            self.pipeline.create_generator(  
                attn_mode="flash_attn2",  
                infer_steps=40,
                num_frames=81,
                guidance_scale=5.0, 
                sample_shift=5.0,
                height=480,
                width=832,
                fps=16,
            )


    def start_turbo(self, name: Literal["wan2.1-turbo", "wan2.1-turbo-fp8"]):
        if name == "wan2.1-turbo":
            model_path = get_path_file_video_model("wan2.1-turbo")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="wan2.1_distill",
                task="t2v",
            )

            self.pipeline.create_generator(  
                infer_steps=4,    
                height=480,  
                width=832,  
                num_frames=81,
                guidance_scale=2.0,
                denoising_step_list=[1000, 750, 500, 250]
            )

        elif name == "wan2.1-turbo-fp8":
            model_path = get_path_file_video_model("wan2.1-turbo-fp8")

            self.pipeline = LightX2VPipeline(
                model_path=model_path,
                model_cls="wan2.1_distill",
                task="t2v",
            )

            self.pipeline.enable_quantize(  
                dit_quantized=True,  
                dit_quantized_ckpt=f"{model_path}/wan2.1_t2v_14b_scaled_fp8_e4m3_lightx2v_4step.safetensors",  
                quant_scheme="fp8-vllm" 
            )  

            self.pipeline.create_generator(  
                infer_steps=4,    
                height=480,  
                width=832,  
                num_frames=81,
                guidance_scale=2.0,
                denoising_step_list=[1000, 750, 500, 250]
            )


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
        
        elif self.model in ["hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p", "hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8", "hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"]:
            self.pipeline = HunyuanVideo_Pipeline(self.model)

        elif self.model in ["wan2.1", "wan2.1-3B", "wan2.1-turbo", "wan2.1-turbo-fp8"]:
            self.pipeline = Wan2_1_Pipeline(self.model)

        return self.pipeline