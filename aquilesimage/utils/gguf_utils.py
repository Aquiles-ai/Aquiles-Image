import importlib
import os
from huggingface_hub import hf_hub_download
from platformdirs import user_data_dir
from pathlib import Path
import logging
from aquilesimage.utils import setup_colored_logger
import json
from datetime import date, datetime
from typing import Optional

logger_p = setup_colored_logger("Aquiles-Image-GGUF-Utils", logging.DEBUG)

registry_file = "registry.json"
hf_repo = "Aquiles-ai/aquiles-gguf-registry"
data_dir = user_data_dir("aquiles", "Aquiles-Image")
os.makedirs(data_dir, exist_ok=True)
AQUILES_GGUF_REGISTRY = os.path.join(data_dir, registry_file)

def import_from_string(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def verify_registry():
    if (Path(AQUILES_GGUF_REGISTRY).exists()):
        logger_p.info("Registry already exists")
    else:
        logger_p.info("Downloading registry")
        hf_hub_download(hf_repo, registry_file, local_dir=data_dir, repo_type="dataset")

def update_registry():
    if Path(AQUILES_GGUF_REGISTRY).exists():
        try:
            os.remove(AQUILES_GGUF_REGISTRY)
            logger_p.info("Registry file removed")
        except OSError as e:
            logger_p.error(f"Failed to remove registry file: {e}")
            return

    try:
        hf_hub_download(hf_repo, registry_file, local_dir=data_dir, repo_type="dataset")
        logger_p.info("Registry downloaded successfully")
    except Exception as e:
        logger_p.error(f"Failed to download registry: {e}")

def save_to_registry(
    model_id: str,
    gguf_repo: str,
    gguf_file: str,
    base_repo: str,
    transformer_cls: str,
    pipeline_cls: str,
    added_by: str,
    registry_path: Optional[str] = None,
    date_added: Optional[str] = None
) -> str:
    target_path = registry_path or AQUILES_GGUF_REGISTRY
    target_path = Path(target_path)

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
        except json.JSONDecodeError as e:
            logger_p.error(f"Failed to parse existing registry at {target_path}: {e}")
            raise
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        registry_data = {}

    registry_data[model_id] = {
        "gguf_repo": gguf_repo,
        "gguf_file": gguf_file,
        "base_repo": base_repo,
        "transformer_cls": transformer_cls,
        "pipeline_cls": pipeline_cls,
        "added_by": added_by,
        "date_added": date_added or date.today().isoformat()
    }

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=4, ensure_ascii=False)
        logger_p.info(f"Model '{model_id}' saved to registry at {target_path}")
    except OSError as e:
        logger_p.error(f"Failed to write registry at {target_path}: {e}")
        raise

    return str(target_path)


if __name__ == "__main__":
    logger_p.info("Tests")
    #verify_registry()
    #update_registry()
    #save_to_registry(
    #    model_id="flux1-dev-q4-1",
    #    gguf_repo="city96/FLUX.1-dev-gguf",
    #    gguf_file="flux1-dev-Q4_1.gguf",
    #    base_repo="black-forest-labs/FLUX.1-dev",
    #    transformer_cls="diffusers.FluxTransformer2DModel",
    #    pipeline_cls="diffusers.FluxPipeline",
    #    added_by="FredyRivera-dev",
    #    registry_path="/home/fredy/projects/Aquiles-Image/aquiles-gguf-registry/registry.json"
    #)