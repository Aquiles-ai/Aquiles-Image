import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.utils.export_utils import export_to_video
import gc

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
        )

        self.pipe.enable_model_cpu_offload()

        if hasattr(self.pipe, 'text_encoder'):
            self.pipe.text_encoder.to('cpu')
        
        self.pipe.enable_attention_slicing(slice_size=1)

        return self.pipe

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def generate(self, prompt: str, **kwargs):
        if self.pipe is None:
            print("X Error: Debes llamar a load() primero")
            return None
        
        default_config = {
            'num_frames': 121,
            'num_inference_steps': 5,
            'guidance_scale': 5.0,
            'height': 704,
            'width': 1280,
        }
        
        config = {**default_config, **kwargs}
        
        self._clear_memory()
        
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                **config,
                output_type="latent"
            )
            
            latents = output.frames[0]
            
            self._clear_memory()

            self.vae.to('cuda')

            chunk_size = 8
            num_chunks = (latents.shape[2] + chunk_size - 1) // chunk_size
            
            decoded_frames = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, latents.shape[2])
                
                print(f"Decodificando frames {start_idx} a {end_idx} de {latents.shape[2]}")
                
                latent_chunk = latents[:, :, start_idx:end_idx, :, :]
                
                with torch.no_grad():
                    decoded_chunk = self.vae.decode(
                        latent_chunk.to('cuda'),
                        return_dict=False
                    )[0]
                    
                    decoded_chunk = decoded_chunk.cpu()
                    decoded_frames.append(decoded_chunk)
                
                self._clear_memory()
                
                print(f"Chunk {i+1}/{num_chunks} completed.")
            
            self.vae.to('cpu')
            self._clear_memory()

            full_video = torch.cat(decoded_frames, dim=2)
            
            video_np = full_video.squeeze(0).permute(1, 2, 3, 0).numpy()
            video_np = (video_np * 255).clip(0, 255).astype('uint8')
            
            print(f"Video decoded: {video_np.shape}")
            
        return video_np

if __name__ == "__main__":
    wp = WanPipelineAPI("Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    pipe = wp.load()

    prompt = "A cat sitting on a beach, watching the sunset"
    frames = wp.generate(prompt)
    
    if frames is not None:
        export_to_video(frames, "output.mp4", fps=24)
        print("Video saved as output.mp4")