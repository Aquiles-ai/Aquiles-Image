import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

class WanPipelineAPI:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.vae: AutoencoderKLWan | None = None
        self.pipe: WanPipeline | None = None

    def load(self):
        """
            This is being designed for the 'Wan-AI/Wan2.2-TI2V-5B-Diffusers' model, so for now we'll only make that one available.
        """
        dtype = torch.float16
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id, 
            subfolder="vae", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.pipe = WanPipeline.from_pretrained(
            self.model_id, 
            vae=self.vae, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        self.pipe.enable_sequential_cpu_offload()

        self.pipe.enable_attention_slicing(slice_size=1)

        self.pipe.enable_vae_slicing()

        return self.pipe