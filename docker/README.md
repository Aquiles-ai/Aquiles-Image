## 🐳 Welcome to the Aquiles-Image Docker folder!

Here you'll find Dockerfiles to deploy Aquiles-Image with GPU support (CUDA 13.0), for both **image generation** and **video generation** models. Prebuilt images are published on Docker Hub, or you can build them yourself from these Dockerfiles.

### Prebuilt images (Docker Hub)

- Image models: [`f4k3r22/aquiles-image`](https://hub.docker.com/r/f4k3r22/aquiles-image)
- Video models: [`f4k3r22/aquiles-video`](https://hub.docker.com/r/f4k3r22/aquiles-video)

```bash
docker pull f4k3r22/aquiles-image:0.7.0   # image models
docker pull f4k3r22/aquiles-video:0.7.0   # video models
```

### Available Dockerfiles

- `Dockerfile.image` — Image generation (diffusers-based models: Flux, StableDiffusion, etc).
- `Dockerfile.video` — Video generation (adds `ffmpeg`, `LightX2V`, `LTX-2`, `bitsandbytes`, `accelerate`).

### Build arguments

Both Dockerfiles support the same build-time arguments:

| Argument | Default | Description |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | Python version installed via `uv python install`. |
| `FROM_SOURCE` | `false` | If `true`, installs `diffusers` and `aquiles-image` directly from GitHub instead of PyPI. |
| `EXTRA_DEPS` | `""` | Space-separated list of extra pip packages to install (e.g. `"wandb accelerate"`). |
| `TORCH_VERSION` | `2.9.0 (Image) and 2.8.0 (Video)` | Pinned PyTorch version. |

> **Note**: `Dockerfile.video` pins the CUDA 13.0 flash-attn wheel to `torch2.9` in its URL. If you override `TORCH_VERSION` to a different patch, update that URL too — the prebuilt wheel is compiled against a specific torch build and won't work otherwise.

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
docker run -p 8000:5500 f4k3r22/aquiles-image:0.7.0 aquiles-image serve --host "0.0.0.0"
```

> Note: replace `f4k3r22/aquiles-image:0.7.0` with `f4k3r22/aquiles-video:0.7.0`, or your own locally built tag (`aquiles-image`/`aquiles-video`), depending on which image you built or pulled. The first argument is the **image name**, the second is the **command** run inside the container.

### Environment variables

Pass `HF_TOKEN` at runtime (never at build time) if you need to pull gated models from Hugging Face:

```bash
docker run -p 8000:5500 -e HF_TOKEN=hf_xxxxx f4k3r22/aquiles-image:0.7.0
```

### Volumes

Both images declare two mount points to keep downloaded models and app data across restarts:

- `/root/.cache/huggingface` — Hugging Face model cache.
- `/root/.local/share` — Aquiles-Image config and generated files.

```bash
docker run -p 8000:5500 \
  -v hf_cache_vol:/root/.cache/huggingface \
  -v aquiles_data_vol:/root/.local/share \
  -e HF_TOKEN=hf_xxxxx \
  f4k3r22/aquiles-image:0.7.0
```

Good luck!