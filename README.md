<div align="center">

# Aquiles-Image

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1763763684/aquiles_image_m6ej7u.png" alt="Aquiles-Image Logo" width="800"/>

### **Self-hosted image generation with OpenAI-compatible APIs**

*🚀 FastAPI • Diffusers • Drop-in replacement for OpenAI*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://platform.openai.com/docs/api-reference/images)
[![PyPI Version](https://img.shields.io/pypi/v/aquiles-image.svg)](https://pypi.org/project/aquiles-image/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/aquiles-image)](https://pypi.org/project/aquiles-image/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Docs-brightgreen.svg)](https://aquiles-ai.github.io/aquiles-image-docs/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Aquiles-ai/Aquiles-Image)
</div>

## 🎯 What is Aquiles-Image?

**Aquiles-Image** is a production-ready API server that lets you run state-of-the-art image generation models on your own infrastructure. OpenAI-compatible by design, you can switch from external services to self-hosted in under 5 minutes.

### Why Aquiles-Image?

| Challenge | Aquiles-Image Solution |
|-----------|------------------------|
| 💸 **Expensive external APIs** | Run models locally with unlimited usage |
| 🔒 **Data privacy concerns** | Your images never leave your server |
| 🐌 **Slow inference** | Advanced optimizations for 3x faster generation |
| 🔧 **Complex setup** | One command to run any supported model |
| 🚫 **Vendor lock-in** | OpenAI-compatible, switch without rewriting code |

### Key Features

- **🔌 OpenAI Compatible** - Use the official OpenAI client with zero code changes
- **⚡ 3x Faster** - Advanced inference optimizations out of the box
- **🎨 10+ Models** - FLUX.1, FLUX.2, SD3.5, and more preconfigured
- **🛠️ Superior DevX** - Simple CLI, dev mode for testing, built-in monitoring
- **🎬 Experimental Video** - Text-to-video generation support (Wan2.2)


## 🚀 Quick Start

### Installation

```bash
# From PyPI (recommended)
pip install aquiles-image

# From source
git clone https://github.com/Aquiles-ai/Aquiles-Image.git
cd Aquiles-Image
pip install .
```

### Launch Server

```bash
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium"
```

### Generate Your First Image

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="not-needed")

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024"
)

print(f"Image URL: {result.data[0].url}")
```

That's it! You're now generating images with the same API you'd use for OpenAI.

## 🎨 Supported Models

### Text-to-Image (`/images/generations`)

- `stabilityai/stable-diffusion-3-medium`
- `stabilityai/stable-diffusion-3.5-medium` 
- `stabilityai/stable-diffusion-3.5-large`
- `stabilityai/stable-diffusion-3.5-large-turbo`
- `black-forest-labs/FLUX.1-dev`
- `black-forest-labs/FLUX.1-schnell`
- `black-forest-labs/FLUX.1-Krea-dev`
- `black-forest-labs/FLUX.2-dev` * 
- `diffusers/FLUX.2-dev-bnb-4bit`
- `Tongyi-MAI/Z-Image-Turbo`

### Image-to-Image (`/images/edits`)

- `black-forest-labs/FLUX.1-Kontext-dev`
- `diffusers/FLUX.2-dev-bnb-4bit`
- `black-forest-labs/FLUX.2-dev` * 

> **\* Note on FLUX.2-dev**: Requires NVIDIA H200 with 64GB RAM minimum. Inference times are variable (17s-2min) and may be unpredictable.

### Text-to-Video (`/videos`) - Experimental

- `Wan-AI/Wan2.2-T2V-A14B` (High quality, 40 steps - requires H100/A100-80G, start with `--model "wan2.2"`)
- `Aquiles-ai/Wan2.2-Turbo` ⚡ **9.5x faster** - Same quality in 4 steps! (requires H100/A100-80G, start with `--model "wan2.2-turbo"`)

> **VRAM Requirements**: Most models need 24GB+ VRAM. Video generation requires 80GB+ (H100/A100-80G).

[**📖 Full models documentation**](https://aquiles-ai.github.io/aquiles-image-docs/#models) and more models in [**🎬 Aquiles-Studio**](https://huggingface.co/collections/Aquiles-ai/aquiles-studio)

## 💡 Examples

### Generating Images

https://github.com/user-attachments/assets/00e18988-0472-4171-8716-dc81b53dcafa

https://github.com/user-attachments/assets/00d4235c-e49c-435e-a71a-72c36040a8d7

### Editing Images

<div align="center">

| Input + Prompt | Result |
|----------------|--------|
| <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1764807968/Captura_de_pantalla_1991_as3v28.png" alt="Edit Script" width="500"/> | <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1764807952/Captura_de_pantalla_1994_ffmko2.png" alt="Edit Result" width="500"/> |

</div>

### Generating Videos (Experimental)

https://github.com/user-attachments/assets/7b1270c3-b77b-48df-a0fe-ac39b2320143

> **Note**: Video generation with `wan2.2` takes ~30 minutes on H100. With `wan2.2-turbo`, it takes only ~3 minutes! Only one video can be generated at a time.

## 🧪 Advanced Features

### AutoPipeline - Run Any Diffusers Model

Run any model compatible with `AutoPipelineForText2Image` from HuggingFace:

```bash
aquiles-image serve \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --auto-pipeline \
  --set-steps 30
```

**Supported models include:**
- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-xl-base-1.0`
- Any HuggingFace model compatible with `AutoPipelineForText2Image`

**Trade-offs:**
- ⚠️ Slower inference than native implementations
- ⚠️ No LoRA or adapter support
- ⚠️ Experimental - may have stability issues

### Dev Mode - Test Without Loading Models

Perfect for development, testing, and CI/CD:

```bash
aquiles-image serve --no-load-model
```

**What it does:**
- Starts server instantly without GPU
- Returns test images that simulate real responses
- All endpoints functional with realistic formats
- Same API structure as production


## 🎯 Use Cases

| Who | What |
|-----|------|
| 🚀 **AI Startups** | Build image generation features without API costs |
| 👨‍💻 **Developers** | Prototype with multiple models using one interface |
| 🏢 **Enterprises** | Scalable, private image AI infrastructure |
| 🔬 **Researchers** | Experiment with cutting-edge models easily |


## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU with 24GB+ VRAM (most models)
- 10GB+ free disk space


## 📚 Documentation

- [**Full Documentation**](https://aquiles-ai.github.io/aquiles-image-docs/)
- [**Client Reference**](https://aquiles-ai.github.io/aquiles-image-docs/#client-api)
- [**Model Guide**](https://aquiles-ai.github.io/aquiles-image-docs/#models)


<div align="center">

**[⭐ Star this project](https://github.com/Aquiles-ai/Aquiles-Image)** • **[🐛 Report issues](https://github.com/Aquiles-ai/Aquiles-Image/issues)**

*Built with ❤️ for the AI community*

</div>