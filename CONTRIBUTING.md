# Contributing to Aquiles-Image

Thank you for your interest in contributing to Aquiles-Image! We appreciate your help in making this project better.

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Reporting Bugs](#reporting-bugs)
- [Adding New Models](#adding-new-models)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)

## Types of Contributions

We currently accept the following types of contributions:

- **Bug reports**: Help us identify and fix issues
- **New image models**: Add support for Diffusers-compatible models

### ⚠️ What we DON'T accept currently

- **New video models**: We are not currently accepting contributions for video models
- **Other libraries**: We only accept models compatible with `diffusers` from HuggingFace

## Reporting Bugs

If you find a bug, please open an [issue](https://github.com/Aquiles-ai/Aquiles-Image/issues) with the following information:

### Required Information

1. **Problem description**: Clearly explain what is failing
2. **Steps to reproduce**: List the exact steps to reproduce the error
3. **Expected behavior**: Describe what you expected to happen
4. **Actual behavior**: Describe what is actually happening
5. **Environment**:
   - Aquiles-Image version
   - Python version
   - Operating system
   - GPU and available VRAM
   - Model you were using
6. **Error logs**: Include the complete traceback if available
7. **Example code**: If possible, include a minimal script that reproduces the error

### Bug Report Template

```markdown
## Bug Description
[Clear and concise description of the problem]

## Steps to Reproduce
1. Run command '...'
2. Call endpoint '...'
3. See error

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What is currently happening]

## Environment
- Aquiles-Image version: [e.g., 0.1.0]
- Python version: [e.g., 3.10.0]
- OS: [e.g., Ubuntu 22.04]
- GPU: [e.g., NVIDIA A100 80GB]
- Model: [e.g., stabilityai/stable-diffusion-3.5-medium]

## Logs
```
[Paste the complete error here]
```

## Example Code (optional)
```python
# Minimal code to reproduce the error
```
```

## Adding New Models

### Prerequisites

Before adding a new model, make sure that:

1. **The model is compatible with Diffusers**: The model must be available on HuggingFace and compatible with `diffusers`
2. **It's an image model**: For now we only accept image models (Text-to-Image or Image-to-Image)
3. **You tested the model locally**: Verify that it works correctly with your implementation
4. **Open an issue first**: For large changes, open an issue describing which model you want to add and why

### Implementation Guide

To add a new model, you need to modify two main files:

#### 1. Implement the Pipeline (`aquilesimage/pipelines/pipelines.py`)

Refer to the [`pipelines.py`](https://github.com/Aquiles-ai/Aquiles-Image/blob/main/aquilesimage/pipelines/pipelines.py) file to understand how pipelines are implemented.

**Basic pipeline structure:**

```python
from diffusers import AutoPipelineForText2Image
import torch
import logging
from aquilesimage.utils import setup_colored_logger

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineYourModel:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: AutoPipelineForText2Image | None = None
        self.device: str | None = None
        self.dist_inf = dist_inf
        self.pipelines = {}  # For distributed inference
    
    def start(self):
        """Initialize and load the model"""
        if torch.cuda.is_available():
            model_path = self.model_path or "organization/default-model"
            logger_p.info("Loading CUDA")
            
            self.device = "cuda"
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
            
            # Call optimizations
            self.optimization()
            
        elif torch.backends.mps.is_available():
            # Support for Mac M-series
            self.device = "mps"
            # Similar implementation
        else:
            raise Exception("No CUDA or MPS device available")
    
    def optimization(self):
        """Apply optimizations to the model"""
        try:
            logger_p.info("Starting optimization process...")
            
            # Torch inductor configuration
            config = torch._inductor.config
            config.conv_1x1_as_mm = True
            config.coordinate_descent_tuning = True
            
            # QKV projection fusion
            self.pipeline.transformer.fuse_qkv_projections()
            self.pipeline.vae.fuse_qkv_projections()
            
            # channels_last memory format
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            
            # FlashAttention
            self.enable_flash_attn()
            
            logger_p.info("All optimizations completed successfully")
        except Exception as e:
            logger_p.error(f"X Error in optimization: {e}")
            raise
    
    def enable_flash_attn(self):
        """Enable FlashAttention if available"""
        try:
            self.pipeline.transformer.set_attention_backend("_flash_3_hub")
            logger_p.info("FlashAttention 3 enabled")
        except Exception as e:
            logger_p.debug(f"FlashAttention 3 not available: {str(e)}")
            try:
                self.pipeline.transformer.set_attention_backend("flash")
                logger_p.info("FlashAttention 2 enabled")
            except Exception as e2:
                logger_p.debug(f"FlashAttention 2 not available: {str(e2)}")
                logger_p.warning("Using default SDPA")
```

**Important tips:**
- Each pipeline is an independent class (does not inherit from a base class)
- Implement `start()` to load and initialize the model
- Implement `optimization()` to apply optimizations (FlashAttention, QKV fusion, etc.)
- Implement `enable_flash_attn()` to attempt to enable FlashAttention
- Use `self.device` for the device (cuda/mps)
- Use `self.pipelines = {}` if you support distributed inference
- Handle exceptions with try-except and appropriate logging
- Follow the pattern of existing classes in `pipelines.py`

**⚠️ Important - Imports with Try-Except:**

If the model you are adding **is only available in the `main` branch of diffusers** (not in a stable release), you must protect the imports with `try-except` blocks to avoid errors on systems using stable versions of diffusers.

**Example of correct imports:**

```python
# At the beginning of the pipelines.py file
try:
    from diffusers.pipelines.your_model.pipeline_your_model import YourModelPipeline
    from diffusers.models.transformers.transformer_your_model import YourModelTransformer
except ImportError as e:
    print("Error import YourModelPipeline")
    pass
```

**Example of protected initialization in the class:**

```python
class PipelineYourModel:
    def __init__(self, model_path: str | None = None, dist_inf: bool = False):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.dist_inf = dist_inf
        try:
            self.pipeline: YourModelPipeline | None = None
        except Exception as e:
            self.pipeline = None
            print("Error import YourModelPipeline")
            pass
        # ... rest of attributes
```

This practice ensures that Aquiles-Image can function even if the specific model is not available in the user's version of diffusers.

#### 2. Update Pydantic Models (`aquilesimage/models/models.py`)

Refer to the [`models.py`](https://github.com/Aquiles-ai/Aquiles-Image/blob/main/aquilesimage/models/models.py) file to see the model structure.

**Add your model to the appropriate `ImageModel` class:**

```python
class ImageModel(str, Enum):
    # ... existing models ...
    YOUR_NEW_MODEL = "organization/model-name"
```

**Classify your model according to its capabilities:**

| Model Type | Base Class | Description |
|------------|------------|-------------|
| **Text-to-Image (T2I)** | `ImageModelBase` | Only generates images from text |
| **Image-to-Image (I2I)** | `ImageModelEdit` | Only edits/modifies existing images |
| **Hybrid (T2I + I2I)** | `ImageModelHybrid` | Can do both functions |

> **📝 Important Note for I2I and Hybrid Models:** If your model has Image-to-Image capabilities, you must specify in your PR whether it supports **single or multiple input images** and what the **optimal maximum number of input images** is. For example: "Supports multi-image editing. Maximum 10 input images" or "Single image input only".

**Classification example:**

```python
# Text-to-Image only
class ImageModelBase(str, Enum):
    SD35_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"
    YOUR_T2I_MODEL = "organization/text-to-image-model"

# Image-to-Image only
class ImageModelEdit(str, Enum):
    FLUX_KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"
    YOUR_I2I_MODEL = "organization/image-to-image-model"

# Hybrid (T2I + I2I)
class ImageModelHybrid(str, Enum):
    FLUX2_DEV = "black-forest-labs/FLUX.2-dev"
    YOUR_HYBRID_MODEL = "organization/hybrid-model"
```

#### 3. Update Documentation

Don't forget to update the README.md by adding your new model to the supported models section, including:
- Model name
- Type (T2I, I2I, or Hybrid)
- Special requirements (VRAM, etc.)
- **For I2I/Hybrid models**: Specify input image support (e.g., "Supports multi-image editing. Maximum 10 input images" or "Single image input only")
- Usage notes if applicable

**Example for I2I models:**
```markdown
### Image-to-Image (`/images/edits`)

- `your-org/your-model-name` - Supports multi-image editing. Maximum 5 input images.
- `your-org/single-edit-model` - Single image input only.
```

#### 4. Add Modal Deployment Example

To facilitate the use of your model, you should include a deployment example on [Modal](https://modal.com) in the `examples/` folder.

**File location:**
```
examples/aquiles_deploy_modelname.py
```

**Example file structure:**

```python
import modal
import os

# Base image with all dependencies
aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential",)
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel"
    )
    .env({"UV_HTTP_TIMEOUT": "600"})
    .uv_pip_install(
        "torch==2.8",
        "git+https://github.com/huggingface/diffusers.git",  # Or specific version if needed
        "transformers==4.57.3",
        "tokenizers==0.22.1",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl",
        "kernels",
        "uvicorn==0.40.0"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1",
          "HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "")})  
)

# Model configuration
MODEL_NAME = "organization/your-model-name"

# Cache volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)

app = modal.App("aquiles-image-server")

# Resource configuration
N_GPU = 1  # Adjust according to model requirements
MINUTES = 60
AQUILES_PORT = 5500

@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"H100:{N_GPU}",  # Adjust according to requirements (A100, H100, etc.)
    scaledown_window=15 * MINUTES, 
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
    },
)
@modal.concurrent(max_inputs=100)
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
        "--set-steps", "30",  # Adjust according to model
        "--api-key", "dummy-api-key"
    ]
    
    print(f"Starting Aquiles-Image with the model:{MODEL_NAME}")
    print(f"Command {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)
```

**Necessary adjustments:**
- `MODEL_NAME`: Use the exact name of your model on HuggingFace
- `gpu`: Specify the type and quantity of GPUs needed (A100, H100, etc.)
- `--set-steps`: Adjust according to model recommendations
- Add additional flags if the model requires them (e.g., `--low-vram`, `--dist-inference`)

**Deployment usage example:**

Users will be able to deploy your model on Modal with:
```bash
modal deploy examples/aquiles_deploy_yourmodel.py
```

And then use it with the OpenAI client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-user--aquiles-image-server-serve.modal.run",
    api_key="dummy-api-key"
)

result = client.images.generate(
    model="organization/your-model-name",
    prompt="a beautiful landscape",
    size="1024x1024"
)
```

Check the [`examples/`](https://github.com/Aquiles-ai/Aquiles-Image/tree/main/examples) folder to see more existing deployment examples.

## Pull Request Process

### Before Creating a PR

1. **Open an issue first** if your contribution is large (new model, significant feature)
2. **Fork** the repository
3. **Create a branch** from `main` with a descriptive name:
   - For bugs: `fix/short-description`
   - For models: `feat/model-name`
   - Example: `feat/stable-diffusion-xl`

### Creating the Pull Request

1. **Descriptive commits**: Write clear commit messages that explain what changes and why
   ```
   Good: "Add support for Stable Diffusion XL model"
   Bad: "Update files"
   
   Good: "Fix memory leak in batch processing"
   Bad: "Fix bug"
   ```

2. **PR description**: Include:
   - What problem does it solve or what does it add?
   - How did you test it?
   - Are there breaking changes?
   - Reference to related issue (if exists)

3. **PR checklist**:
   - [ ] The code works correctly
   - [ ] I tested the model/fix locally
   - [ ] I updated the documentation (README.md)
   - [ ] I added a Modal deployment example (`examples/` folder)
   - [ ] Imports use try-except if the model is only in `main` of diffusers
   - [ ] Commits are descriptive
   - [ ] I followed the existing file structure
   - [ ] **For I2I/Hybrid models**: I specified if it supports single/multiple input images and the optimal maximum

### Pull Request Template

```markdown
## Description
[Describe what this PR adds or solves]

## Type of change
- [ ] Bug fix
- [ ] New image model
- [ ] Documentation update

## Model Information (if applicable)
- Model type: [T2I / I2I / Hybrid]
- **For I2I/Hybrid models**: 
  - Input images support: [Single image / Multiple images (max: X)]
  - Optimal maximum input images: [e.g., 10, 5, 3, or N/A for single image]

## How was it tested?
[Describe how you tested the changes]

## Checklist
- [ ] The code works correctly
- [ ] I tested locally
- [ ] I updated the documentation
- [ ] I added a Modal deployment example (if new model)
- [ ] Imports use try-except (if model only in `main` of diffusers)
- [ ] Descriptive commits
- [ ] I followed the project structure
- [ ] **For I2I/Hybrid models**: Specified input image support and optimal maximum

## Related issue
Fixes #[issue number]
```

## Code Standards

For now we maintain basic standards:

### Commits

- **Descriptive**: Explain what you change and why
- **In English**: Commits must be in English
- **Suggested format**: 
  ```
  Add support for [model-name]
  Fix [specific issue] in [component]
  Update [file] to include [change]
  ```

### Code Structure

- Follow the existing file structure
- Maintain consistency with current code
- Comment complex sections when necessary

## Questions?

If you have any questions about how to contribute, don't hesitate to:

1. Open a [discussion issue](https://github.com/Aquiles-ai/Aquiles-Image/issues)
2. Review existing issues to see if your question has already been answered

Thank you for contributing to Aquiles-Image! 🚀