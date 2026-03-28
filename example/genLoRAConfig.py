from aquilesimage.utils import save_lora_config
from aquilesimage.models import LoRAConfig

# Save the LoRA configuration to a JSON file (FLUX.1-dev is the base model)

save_lora_config(
    LoRAConfig(
        repo_id="brushpenbob/Flux-retro-Disney-v2",
        weight_name="Flux_retro_Disney_v2.safetensors",
        adapter_name="flux-retro-disney-v2"
    ),
    "./lora_config.json"
)