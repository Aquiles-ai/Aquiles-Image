from aquilesimage.models import LoRAConfig
from diffusers import DiffusionPipeline
import logging
from aquilesimage.utils import setup_colored_logger

logger = setup_colored_logger("Aquiles-LoRA-Loader", logging.INFO)

def loadLoRA(pipeline: DiffusionPipeline, conf: LoRAConfig):
    try:
        if conf.prefix is None:
            pipeline.load_lora_weights(
                conf.repo_id,
                weight_name=conf.weight_name,
                adapter_name=conf.adapter_name,
            )
        else:
            component = getattr(pipeline, conf.prefix, None)

            if component is None:
                logger.error(f"X Pipeline has no component '{conf.prefix}'.")
                logger.info("There was an error loading LoRA. Only the base model is loaded.")
                return False

            component.load_lora_adapter(
                conf.repo_id,
                weight_name=conf.weight_name,
                adapter_name=conf.adapter_name,
                prefix=conf.prefix,
            )

        logger.info(f"LoRA '{conf.adapter_name}' loaded successfully from '{conf.repo_id}'.")
        return True

    except Exception as e:
        logger.error(f"X Error loading LoRA: {e}")
        logger.info("There was an error loading LoRA. Only the base model is loaded.")
        return False