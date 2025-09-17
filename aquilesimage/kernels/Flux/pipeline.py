import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from aquilesimage.kernels.Flux.image_processor import PipelineImageInput, VaeImageProcessor
from aquilesimage.kernels.Flux.schedulers import FlowMatchEulerDiscreteScheduler
from aquilesimage.kernels.models.transformersflux import FluxTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.loaders.ip_adapter import FluxIPAdapterMixin
from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
from diffusers.loaders.single_file import FromSingleFileMixin
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.utils import logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.import_utils import is_torch_xla_available
from diffusers.utils.doc_utils import replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from aquilesimage.kernels.Flux.pipeline_output import FluxPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""