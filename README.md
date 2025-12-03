<div align="center">

# Aquiles-Image

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1763763684/aquiles_image_m6ej7u.png" alt="Aquiles-Image Logo" width="2560" height="1200"/>

### **Easy, fast and cheap Diffusion Models that work for everyone.**

*🚀 FastAPI • Diffusers • Compatible with the OpenAI client*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://platform.openai.com/docs/api-reference/images)
[![PyPI Version](https://img.shields.io/pypi/v/aquiles-image.svg)](https://pypi.org/project/aquiles-image/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/aquiles-image)](https://pypi.org/project/aquiles-image/)


</div>

## 🔥 What's New in Aquiles-Image

<div align="center">

| Feature | Description |
|---------|-------------|
| ⚡ **3x Faster** | Advanced inference optimizations |
| 🎨 **More Models** | Support for FLUX, SD3-3.5, Flux2 and more |
| 🔧 **Better DevX** | Improved CLI and monitoring capabilities |
| 🔌 **OpenAI Compatible** | Drop-in replacement for OpenAI's image APIs  |

</div>

## 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU with 24GB+ VRAM
- 10GB+ free disk space

## Generating an image with `stabilityai/stable-diffusion-3.5-medium`

https://github.com/user-attachments/assets/00e18988-0472-4171-8716-dc81b53dcafa

## Generating an image with `black-forest-labs/FLUX.1-Krea-dev`



https://github.com/user-attachments/assets/00d4235c-e49c-435e-a71a-72c36040a8d7



## ⚙️ Installation

### From Pypi
```bash
uv pip install aquiles-image
```
### From source
```bash
git clone https://github.com/Aquiles-ai/Aquiles-Image.git
cd Aquiles-Image
uv pip install .
```

## 🚀 Launch your Aquiles-Image server

```bash
aquiles-image serve --host "0.0.0.0" --port 5500 --model "stabilityai/stable-diffusion-3.5-medium"
```

## 🎨 Supported Models

<div align="center">

| Model | Endpoint |
|-------|----------|
| [`stabilityai/stable-diffusion-3-medium`](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | `/images/generations` |
| [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | `/images/generations` |
| [`stabilityai/stable-diffusion-3.5-large`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | `/images/generations` |
| [`stabilityai/stable-diffusion-3.5-large-turbo`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) | `/images/generations` |
| [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `/images/generations` |
| [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `/images/generations` |
| [`black-forest-labs/FLUX.1-Krea-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) | `/images/generations` |
| [`diffusers/FLUX.2-dev-bnb-4bit`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) | `/images/generations` |
| [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev) | `/images/generations` |
| [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `/images/generations` |
| [`black-forest-labs/FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | `/images/edits` |

</div>

> ⚠️ **VRAM Requirements**: Most models require 24GB+ VRAM with an additional ~10GB free for processing.

## 💻 Start your Aquiles-Image server in dev mode without loading models

Dev mode allows you to start the server without loading any AI models, ideal for rapid development, integration testing, or endpoint validation without requiring GPU or heavy computational resources.

```bash
aquiles-image serve --host "0.0.0.0" --port 5500 --no-load-model
```

### What does dev mode do?

- **No model loading**: Server starts instantly without downloading or loading AI models
- **Functional endpoints**: All endpoints respond normally with test images
- **Realistic responses**: Returns valid images that simulate model responses
- **Same format**: Responses maintain the exact API format (URLs, base64, metadata)

### Use cases

- API integration development  
- Endpoint testing without GPU  
- Workflow validation  
- CI/CD environment testing  
- Development on machines without GPU resources  

> **Note**: Dev mode is for development only. For production, use the normal server with loaded models.

## 🎉 Generate your first image with Aquiles-Image

```py
from openai import OpenAI
import requests

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024"
)

print(f"URL of the generated image: {result.data[0].url}\n")

image_url = result.data[0].url
response = requests.get(image_url)

with open("image.png", "wb") as f:
    f.write(response.content)

print(f"Image downloaded successfully\n")
```

## 🎯 Perfect For

<div align="center">

| Use Case | Description |
|----------|-------------|
| 🚀 **AI Startups** | Building image generation features |
| 👨‍💻 **Developers** | Prototyping with Image Generation Models |
| 🏢 **Enterprises** | Scalable image AI infrastructure |
| 🔬 **Researchers** | Experimenting with multiple models  |

</div>

<div align="center">

*Built with ❤️ for the AI community*

**[⭐ Star this project](https://github.com/Aquiles-ai/Aquiles-Image) • [📖 Documentation](#) • [💬 Community](#)**

</div>
