## 🐳 Welcome to the Aquiles-Image Docker folder!

Here you'll find Dockerfiles to deploy Aquiles-Image with GPU support (CUDA 13.0), for both **image generation** and **video generation** models. Prebuilt images are published on Docker Hub, or you can build them yourself from these Dockerfiles.

### Prebuilt images (Docker Hub)

- Image models: [`f4k3r22/aquiles-image`](https://hub.docker.com/r/f4k3r22/aquiles-image)
- Video models: [`f4k3r22/aquiles-video`](https://hub.docker.com/r/f4k3r22/aquiles-video)

```bash
docker pull f4k3r22/aquiles-image:latest   # image models
docker pull f4k3r22/aquiles-video:latest   # video models
```

### Available Dockerfiles

- `Dockerfile.image` — Image generation (diffusers-based models: Flux, StableDiffusion, etc).
- `Dockerfile.video` — Video generation (adds `ffmpeg`, `LightX2V`, `LTX-2`, `bitsandbytes`, `accelerate`, `vLLM`).

### Build arguments

Both Dockerfiles share these build-time arguments:

| Argument | Default | Description |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | Python version installed via `uv python install`. |
| `FROM_SOURCE` | `false` | If `true`, installs `diffusers` and `aquiles-image` directly from GitHub instead of PyPI. |
| `EXTRA_DEPS` | `""` | Space-separated list of extra pip packages to install (e.g. `"wandb accelerate"`). |
| `TORCH_VERSION` | `2.9` (Image), `2.13.0` (Video) | Pinned PyTorch version. |
| `AQUILES_VERSION` | `0.8.5` | `aquiles-image` version installed from PyPI when `FROM_SOURCE=false`. |

`Dockerfile.video` adds these arguments:

| Argument | Default | Description |
|---|---|---|
| `TORCHVISION_VERSION` | `0.28.0` | `torchvision` version matching `TORCH_VERSION`. |
| `TRANSFORMERS_VERSION` | `5.19.0` | Must stay below `5.20` while vLLM 0.32 is pinned. |
| `VLLM_VERSION` | `0.32.0` | vLLM version. Requires `torch==2.13.0`. |

> **Note**: `Dockerfile.video` installs `torch` from the CUDA 13.0 PyTorch index and pins the flash-attn wheel to `torch2.13` (release `v0.9.47`). Keep `TORCH_VERSION`, `TORCHVISION_VERSION`, `TRANSFORMERS_VERSION`, `VLLM_VERSION` and the flash-attn URL in sync. The wheel is compiled against a specific torch build, and vLLM 0.32 only accepts `transformers<5.20`. Overriding one without the others will break the build or the runtime.

### How to build

```bash
# Image model, PyPI packages, default Python
docker build -f Dockerfile.image -t aquiles-image .

# Video model, from source, extra deps
docker build -f Dockerfile.video \
  --build-arg FROM_SOURCE=true \
  --build-arg EXTRA_DEPS="wandb" \
  -t aquiles-video .
```

### How to run

The entrypoint accepts any `aquiles-image` CLI command. If you don't pass anything, it defaults to `aquiles-image serve`.

```bash
docker run -p 8000:5500 --gpus all f4k3r22/aquiles-image:latest aquiles-image serve --host "0.0.0.0"
```

### Environment variables

Pass `HF_TOKEN` at runtime (never at build time) if you need to pull gated models from Hugging Face:

```bash
docker run -p 8000:5500 --gpus all -e HF_TOKEN=hf_xxxxx f4k3r22/aquiles-image:latest
```

### Volumes

Both images declare two mount points to keep downloaded models and app data across restarts:

- `/root/.cache/huggingface` — Hugging Face model cache.
- `/root/.local/share` — Aquiles-Image config and generated files.

```bash
docker run -p 8000:5500 \
  --gpus all \
  -v hf_cache_vol:/root/.cache/huggingface \
  -v aquiles_data_vol:/root/.local/share \
  -e HF_TOKEN=hf_xxxxx \
  f4k3r22/aquiles-image:latest
```

Good luck!