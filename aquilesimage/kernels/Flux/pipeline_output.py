from dataclasses import dataclass
from typing import List, Union
import numpy as np
from PIL.Image import Image
import torch
from diffusers.utils.outputs import BaseOutput

@dataclass
class FluxPipelineOutput(BaseOutput):
    """
    Output class for Flux image generation pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: Union[List[Image], np.ndarray]