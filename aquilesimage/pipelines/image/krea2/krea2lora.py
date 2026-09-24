import torch
from aquilesimage.utils import setup_colored_logger
import logging
from aquilesimage.models import BasePipeline
from aquilesimage.utils import _lora_conf_krea2
import inspect

logger_p = setup_colored_logger("Aquiles-Image-Pipelines", logging.DEBUG)

_KREA2_AVAILABLE = False
try:
    from diffusers import Krea2Pipeline
    _KREA2_AVAILABLE = True
except ImportError as e:
    logger_p.info("Error import Krea2Pipeline")
    pass

if _KREA2_AVAILABLE:
    class Krea2PipelineWithLoRA(Krea2Pipeline):

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            lora: str | None = kwargs.pop("lora", None)

            pipeline = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            pipeline._lora_name = lora

            return pipeline

        def _maybe_append_trigger(self, prompt: str, trigger: str) -> str:
            if not isinstance(prompt, str) or not trigger:
                return prompt
            # Avoid duplicating the trigger if the user already included it.
            if trigger.lower() in prompt.lower():
                return prompt
            return f"{prompt}, {trigger}"

        def __call__(self, *args, **kwargs):
            if args:
                sig = inspect.signature(Krea2Pipeline.__call__)
                params = list(sig.parameters.keys())[1:]  # skip 'self'
                for i, val in enumerate(args):
                    if i < len(params):
                        kwargs[params[i]] = val
                args = ()

            lora_name = getattr(self, "_lora_name", None)
            if lora_name and lora_name in _lora_conf_krea2:
                trigger = _lora_conf_krea2[lora_name]["trigger"]
                prompt = kwargs.get("prompt")

                if isinstance(prompt, list):
                    kwargs["prompt"] = [self._maybe_append_trigger(p, trigger) for p in prompt]
                elif isinstance(prompt, str):
                    kwargs["prompt"] = self._maybe_append_trigger(prompt, trigger)
                elif prompt is None and kwargs.get("prompt_embeds") is not None:
                    logger_p.debug("prompt_embeds provided, LoRA trigger cannot be injected into embeddings.")

            return super().__call__(*args, **kwargs)
else:
    Krea2PipelineWithLoRA = None

class PipelineKrea2LoRA(BasePipeline):
    def __init__(self, model_path: str | None = None, dist_inf: bool = False,
                 lora_scale: float = 1.0):
        self.model_name = "krea/Krea-2-Turbo"
        self.lora_name = model_path
        self.lora_scale = float(lora_scale) if lora_scale is not None else 1.0
        self.pipeline: Krea2PipelineWithLoRA | None = None

    def start(self):
        if not _KREA2_AVAILABLE or Krea2PipelineWithLoRA is None:
            raise ImportError("Krea2Pipeline is not available in this diffusers version.")

        if not self.lora_name:
            raise ValueError("A Krea2 LoRA model id is required (e.g. 'krea/Krea-2-LoRA-retroanime').")

        if self.lora_name not in _lora_conf_krea2:
            available = ", ".join(sorted(_lora_conf_krea2.keys()))
            raise ValueError(
                f"Unknown Krea2 LoRA '{self.lora_name}'. Available: {available}."
            )

        from diffusers.quantizers import DiffusersAutoQuantizer

        original_from_config = DiffusersAutoQuantizer.from_config
        DiffusersAutoQuantizer.from_config = classmethod(lambda cls, *args, **kwargs: None)

        try:
            self.pipeline = Krea2PipelineWithLoRA.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                lora=self.lora_name
            ).to("cuda")
        finally:
            DiffusersAutoQuantizer.from_config = original_from_config

        wg = _lora_conf_krea2[self.lora_name]["weight_name"]

        # Official Krea usage:
        # pipe.transformer.load_lora_adapter(repo, weight_name=...)
        # pipe.transformer.set_adapters("default", weights=1.0)
        # Note: at transformer level (PeftAdapterMixin) the kwarg is
        # `weights`, not `adapter_weights`.
        self.pipeline.transformer.load_lora_adapter(
            self.lora_name, weight_name=wg, adapter_name="default"
        )

        self.pipeline.transformer.set_adapters("default", weights=self.lora_scale)

        self.optimization()


    def optimization(self):
        try:
            logger_p.info("Skip QKV projections fused & Channels last memory format enabled")
            # QKV fusion is skipped on purpose: fusing after LoRA injection
            # can break or silence the adapter (to_q/k/v/out targets).
            #logger_p.info("QKV projections fused")
            #self.pipeline.transformer.fuse_qkv_projections()
            #self.pipeline.vae.fuse_qkv_projections()
            #logger_p.info("Channels last memory format enabled")
            #self.pipeline.transformer.to(memory_format=torch.channels_last)
            #self.pipeline.vae.to(memory_format=torch.channels_last)
            try:
                logger_p.info("FlashAttention")
                self.enable_flash_attn()
            except Exception as ea:
                logger_p.warning(f"Error in optimization (flash_attn): {str(ea)}")
                pass
        except Exception as e:
            logger_p.warning(f"Error in optimization: {str(e)}")
            pass
