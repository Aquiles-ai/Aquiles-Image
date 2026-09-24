from aquilesimage.models import LoRAConfig
from diffusers import DiffusionPipeline
import logging
from aquilesimage.utils import setup_colored_logger

logger = setup_colored_logger("Aquiles-LoRA-Loader", logging.INFO)


def _resolve_scale(conf: LoRAConfig) -> float:
    try:
        scale = conf.scale if conf.scale is not None else 1.0
    except Exception:
        return 1.0
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        logger.warning(f"Invalid LoRA scale '{conf.scale}', falling back to 1.0.")
        return 1.0
    return scale


def _activate_pipeline_adapter(pipeline: DiffusionPipeline, adapter_name: str, scale: float) -> None:
    # LoraBaseMixin.set_adapters(adapter_names, adapter_weights=...).
    # A single loaded adapter is usually auto-active, but activating
    # explicitly guarantees the LoRA is used and applies LoRAConfig.scale.
    # See: https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference
    if not hasattr(pipeline, "set_adapters"):
        logger.warning("Pipeline has no 'set_adapters', skipping explicit LoRA activation.")
        return
    try:
        pipeline.set_adapters(adapter_name, adapter_weights=scale)
    except TypeError:
        # Older diffusers revisions used `weights` instead of `adapter_weights`.
        pipeline.set_adapters(adapter_name, weights=scale)
    active = None
    try:
        if hasattr(pipeline, "get_active_adapters"):
            active = pipeline.get_active_adapters()
    except Exception:
        active = None
    if active is not None:
        logger.info(f"Active LoRA adapters: {active}.")
    else:
        logger.info(f"LoRA '{adapter_name}' activated with scale={scale}.")


def _activate_component_adapter(component, adapter_name: str, scale: float) -> None:
    # PeftAdapterMixin.set_adapters(adapter_names, weights=...) at model level.
    # See: https://huggingface.co/docs/diffusers/main/en/api/loaders/peft
    if not hasattr(component, "set_adapters"):
        logger.warning("LoRA component has no 'set_adapters', skipping explicit activation.")
        return
    try:
        component.set_adapters(adapter_name, weights=scale)
    except TypeError:
        component.set_adapters(adapter_name, adapter_weights=scale)
    logger.info(f"LoRA '{adapter_name}' activated on component with scale={scale}.")


def loadLoRA(pipeline: DiffusionPipeline, conf: LoRAConfig):
    try:
        if conf is None:
            logger.error("LoRA config is None.")
            logger.info("There was an error loading LoRA. Only the base model is loaded.")
            return False

        scale = _resolve_scale(conf)

        if conf.prefix is None:
            pipeline.load_lora_weights(
                conf.repo_id,
                weight_name=conf.weight_name,
                adapter_name=conf.adapter_name,
            )
            _activate_pipeline_adapter(pipeline, conf.adapter_name, scale)
        else:
            component = getattr(pipeline, conf.prefix, None)

            if component is None:
                logger.error(f"Pipeline has no component '{conf.prefix}'.")
                logger.info("There was an error loading LoRA. Only the base model is loaded.")
                return False

            if not hasattr(component, "load_lora_adapter"):
                logger.error(f"Component '{conf.prefix}' does not support 'load_lora_adapter'.")
                logger.info("There was an error loading LoRA. Only the base model is loaded.")
                return False

            component.load_lora_adapter(
                conf.repo_id,
                weight_name=conf.weight_name,
                adapter_name=conf.adapter_name,
                prefix=conf.prefix,
            )
            _activate_component_adapter(component, conf.adapter_name, scale)

        logger.info(f"LoRA '{conf.adapter_name}' loaded successfully from '{conf.repo_id}'.")
        return True

    except Exception as e:
        logger.error(f"Error loading LoRA: {e}")
        logger.info("There was an error loading LoRA. Only the base model is loaded.")
        return False
