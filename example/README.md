## 🤗 Welcome to the Aquiles-Image examples folder!

Here you'll find examples of how to deploy Aquiles-Image on [Modal](https://modal.com) and use it with the OpenAI client.

### Folder layout

Each subfolder holds ready-to-use Modal deployment files (`aquiles_deploy_*.py`), one per model or setup. Pick the file matching what you want to serve and deploy it as-is, or copy it as a starting point:

- **`image-models/`**: deploys image models (text-to-image, image editing, LoRA, GGUF, quantized and distributed setups). Examples: Flux, Stable Diffusion 3.5, Qwen-Image, Z-Image, GLM, Krea 2, Ideogram 4, Ernie, Nucleus.
- **`video-models/`**: deploys video models (text-to-video and image-to-video). Examples: LTX-2, Wan 2.1/2.2, HunyuanVideo-1.5.

Each deployment file already sets the Modal image (CUDA, torch, diffusers, Aquiles-Image), the `MODEL_NAME`, the GPU type, and the `aquiles-image serve` command, so you only need to run it.

### How to deploy on Modal?

To deploy on Modal, first go to the [Modal website](https://modal.com) and create an account.

Then set it up on your local machine with these commands:
```bash
# Install Modal SDK
uv pip install modal

# Authenticate
python3 -m modal setup
```

Since we have Modal files ready for deployment, simply run this command with the deployment file you want to use whether it's Flux, StableDiffusion, etc. All files come pre-configured and ready to work. Good luck!

**Command**
```bash
# Replace with the deployment file you'll use
modal deploy aquiles_deploy_*.py
```

### Using the OpenAI client

Once you have Aquiles-Image deployed on Modal, you'll have access to a URL like: `https://your-user--aquiles-image-server-serve.modal.run`. You can use this with the `openai_*.py` files to generate/edit images, create videos, etc. (capabilities depend on the model you deployed).

> Note: You can use any OpenAI SDK as long as it supports the `Image API`.
