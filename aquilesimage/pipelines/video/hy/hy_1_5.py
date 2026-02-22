import torch
try:
    from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
    from lightx2v import LightX2VPipeline
except ImportError as e:
    print("Error importing components for LightX2VPipeline")
    pass
from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_hy
from typing import Literal

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