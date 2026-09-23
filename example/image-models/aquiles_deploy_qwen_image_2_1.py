import modal

aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", 
        "curl", 
        "build-essential", 
        "wget")
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel",
    )
    .uv_pip_install(
        "torch==2.12",
        "git+https://github.com/huggingface/diffusers",
        "transformers==5.17.0",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
        "bitsandbytes",
        "accelerate",
        "av"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  
)

MODEL_NAME = "Qwen/Qwen-Image-2.1"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)

app = modal.App("aquiles-image-server")
 
N_GPU = 1
MINUTES = 60
AQUILES_PORT = 5500

@app.function(
    image=aquiles_image,
    gpu=f"H100:{N_GPU}",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=30 * MINUTES, 
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=AQUILES_PORT, startup_timeout=20 * MINUTES)
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
        "--api-key", "dummy-api-key",
        "--username", "root", 
        "--password", "root",
        "--set-steps", "40"
    ]

    print(f"Starting Aquiles-Image with the model:{MODEL_NAME}")
    print(f"Command {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)