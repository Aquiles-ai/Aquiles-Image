import importlib
import logging
import torch
from diffusers import GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
from aquilesimage.models import BasePipeline, LoRAConfig
from aquilesimage.runtime import loadLoRA
from aquilesimage.utils import setup_colored_logger
from aquilesimage.utils.gguf_utils import import_from_string

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

class PipelineGGUFAuto(BasePipeline):
    def __init__(self, entry: dict, low_vram: bool = False, 
            load_lora: bool = False, conf_lora: LoRAConfig | None = None):
        self.entry = entry
        self.low_vram = low_vram
        self.load_lora = load_lora
        self.conf_lora = conf_lora
        self.pipeline = None
        self.device: str | None = None
 
        required = {"gguf_repo", "gguf_file", "base_repo", "transformer_cls", "pipeline_cls"}
        missing = required - self.entry.keys()
        if missing:
            raise ValueError(f"Incomplete registry entry, missing fields: {missing}")

    def start(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            raise RuntimeError("No CUDA or MPS device available")
 
        logger_p.info(
            f"Downloading GGUF: {self.entry['gguf_repo']}/{self.entry['gguf_file']}"
        )
        gguf_path = hf_hub_download(
            repo_id=self.entry["gguf_repo"],
            filename=self.entry["gguf_file"],
        )
 
        TransformerCls = import_from_string(self.entry["transformer_cls"])
        PipelineCls = import_from_string(self.entry["pipeline_cls"])
 
        logger_p.info(
            f"Loading GGUF transformer with {self.entry['transformer_cls']}..."
        )
        transformer = TransformerCls.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=self.entry["base_repo"],
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
 
        logger_p.info(
            f"Assembling pipeline with {self.entry['pipeline_cls']} from {self.entry['base_repo']}..."
        )
        self.pipeline = PipelineCls.from_pretrained(
            self.entry["base_repo"],
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
 
        if self.load_lora:
            loadLoRA(self.pipeline, self.conf_lora)
 
        self.optimization()

    def optimization(self):
        logger_p.info("Applying GGUF-compatible optimizations...")
 
        # enable_sequential_cpu_offload no es compatible con GGUF
        if self.low_vram:
            self.pipeline.enable_model_cpu_offload()

        logger_p.info("GGUF optimizations applied (cpu_offload only)")