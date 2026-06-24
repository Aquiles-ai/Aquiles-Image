import torch
from aquilesimage.utils import setup_colored_logger
import logging
from aquilesimage.models import LoRAConfig
from aquilesimage.runtime import loadLoRA
from aquilesimage.models import BasePipeline

class PipelineKrea2(BasePipeline):
    def __init__(self, model_path: str | None = None, dist_inf: bool = False,
                load_lora: bool = False, conf_lora: LoRAConfig | None = None):
        self.model_name = model_path