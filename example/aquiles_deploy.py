import modal
import os

aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential",)
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel"
    )
    .uv_pip_install(
        "torch==2.11",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==5.14.1",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl",
        "kernels",
        "rich"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  
)

MODEL_NAME = "stabilityai/stable-diffusion-3.5-medium"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)

app = modal.App("aquiles-image-server")

N_GPU = 1
MINUTES = 60
AQUILES_PORT = 5500

@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"L40S:{N_GPU}",
    scaledown_window=15 * MINUTES, 
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
    },
)
@modal.concurrent(max_inputs=1000)
@modal.web_server(port=AQUILES_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "aquiles-image",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(AQUILES_PORT),
        "--model",
        MODEL_NAME,
        "--set-steps", "30",
        "--api-key", "dummy-api-key"
    ]

    print(f"Starting Aquiles-Image with the model:{MODEL_NAME}")
    print(f"Command {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)