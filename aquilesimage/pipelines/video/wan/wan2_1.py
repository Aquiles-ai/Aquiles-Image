import torch
try:
    from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
    from lightx2v import LightX2VPipeline
except ImportError as e:
    print("Error importing components for LightX2VPipeline")
    pass
from aquilesimage.utils.utils_video import get_path_file_video_model, file_exists, download_wan2_1
from typing import Literal

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