<div align="center">

# Aquiles-Image

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1762730973/aquilesim_ow8kbf.png" alt="Aquiles-Image Logo" width="280"/>

### **Easy, fast and cheap Diffusion Models that work for everyone.**

*🚀 FastAPI • Diffusers • Compatible with the OpenAI client*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://platform.openai.com/docs/api-reference/images)


</div>

## 🔥 What's New in Aquiles-Image

<div align="center">

| Feature | Description |
|---------|-------------|
| ⚡ **3x Faster** | Advanced inference optimizations |
| 🎨 **More Models** | Support for FLUX, SD3-3.5, Qwen-Image and more |
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

> Note on model compatibility

> ⚠️ Keep in mind that many of these models require a lot of VRAM to run, choose the model that fits your GPU and has about 10GB left to avoid errors when decoding or saving.

> The supported models for the imaging endpoint are: `stabilityai/stable-diffusion-3-medium`, `stabilityai/stable-diffusion-3.5-large`, `stabilityai/stable-diffusion-3.5-large-turbo`, `stabilityai/stable-diffusion-3.5-medium`, `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell`, `black-forest-labs/FLUX.1-Krea-dev`, `Qwen/Qwen-Image`

> For Edit endpoints (Available, but with some errors that do not guarantee full endpoint functionality): `black-forest-labs/FLUX.1-Kontext-dev`, `Qwen/Qwen-Image-Edit`

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
