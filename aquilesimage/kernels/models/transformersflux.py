import inspect
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from aquilesimage.kernels.configs import ConfigMixin, register_to_config
from diffusers.loaders.transformer_flux import FluxTransformer2DLoadersMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.utils import logging
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

logger = logging.get_logger(__name__)



@dataclass
class EmbeddingCacheKey:
    ids_hash: str
    timestep: Optional[float]
    guidance: Optional[float]
    pooled_proj_hash: Optional[str]
    
    @classmethod
    def from_inputs(cls, ids: torch.Tensor, timestep: Optional[torch.Tensor] = None, 
                   guidance: Optional[torch.Tensor] = None, 
                   pooled_projections: Optional[torch.Tensor] = None):
        ids_hash = hashlib.md5(ids.cpu().numpy().tobytes()).hexdigest()
        
        timestep_val = timestep.item() if timestep is not None else None
        guidance_val = guidance.item() if guidance is not None else None
        
        pooled_hash = None
        if pooled_projections is not None:
            pooled_hash = hashlib.md5(pooled_projections.cpu().numpy().tobytes()).hexdigest()
        
        return cls(
            ids_hash=ids_hash,
            timestep=timestep_val, 
            guidance=guidance_val,
            pooled_proj_hash=pooled_hash
        )


@dataclass
class CachedEmbedding:
    embedding: torch.Tensor
    device: str
    dtype: str
    
    def to_device(self, target_device: torch.device, target_dtype: torch.dtype) -> torch.Tensor:
        if self.device != str(target_device) or self.dtype != str(target_dtype):
            return self.embedding.to(device=target_device, dtype=target_dtype)
        return self.embedding


class EmbeddingCache:    
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache: Dict[EmbeddingCacheKey, CachedEmbedding] = {}
        self.access_order: List[EmbeddingCacheKey] = []
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: EmbeddingCacheKey) -> Optional[CachedEmbedding]:
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: EmbeddingCacheKey, value: CachedEmbedding):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate
            }

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.warning("Flash Attention not available. Using standard attention.")


class OptimizedFluxAttnProcessor:
    _attention_backend = "flash_attn" if FLASH_ATTENTION_AVAILABLE else None

    def __init__(self, use_flash_attn: bool = True, enable_kv_cache: bool = False):
        self.use_flash_attn = use_flash_attn and FLASH_ATTENTION_AVAILABLE
        self.enable_kv_cache = enable_kv_cache
        self.kv_cache = {} 
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

    def _get_qkv_projections(self, attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            encoder_query = encoder_key = encoder_value = (None,)
            if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
                encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)  
            value = attn.to_v(hidden_states)

            encoder_query = encoder_key = encoder_value = None
            if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)

        return query, key, value, encoder_query, encoder_key, encoder_value

    def _flash_attention_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_flash_attn:
            return dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend)
        
        try:
            batch_size, seq_len, num_heads, head_dim = query.shape

            causal = False
            
            output = flash_attn_func(
                query, key, value,
                dropout_p=0.0,
                softmax_scale=None,  
                causal=causal,
                window_size=(-1, -1), 
                deterministic=False
            )
            
            return output
            
        except Exception as e:
            logger.warning(f"Flash Attention failed: {e}. Falling back to standard attention.")
            return dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=None)

    def __call__(
        self,
        attn: "FluxAttention", 
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        query, key, value, encoder_query, encoder_key, encoder_value = self._get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))  
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = self._flash_attention_forward(query, key, value, attention_mask)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class OptimizedFluxIPAdapterAttnProcessor(torch.nn.Module):
    _attention_backend = "flash_attn" if FLASH_ATTENTION_AVAILABLE else None

    def __init__(
        self, hidden_size: int, cross_attention_dim: int, num_tokens=(4,), scale=1.0, device=None, dtype=None,
        use_flash_attn: bool = True
    ):
        super().__init__()
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.use_flash_attn = use_flash_attn and FLASH_ATTENTION_AVAILABLE

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList([
            nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
            for _ in range(len(num_tokens))
        ])
        self.to_v_ip = nn.ModuleList([
            nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
            for _ in range(len(num_tokens))
        ])

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        main_processor = OptimizedFluxAttnProcessor(use_flash_attn=self.use_flash_attn)
        
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = main_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )
            
            if ip_hidden_states:
                ip_attn_output = torch.zeros_like(hidden_states)
                
                query = attn.to_q(hidden_states)
                query = query.unflatten(-1, (attn.heads, -1))
                query = attn.norm_q(query)
                
                for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                    ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
                ):
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                    if self.use_flash_attn:
                        current_ip_hidden_states = main_processor._flash_attention_forward(
                            query, ip_key, ip_value, None
                        )
                    else:
                        current_ip_hidden_states = dispatch_attention_fn(
                            query, ip_key, ip_value, attn_mask=None, backend=self._attention_backend
                        )
                    
                    current_ip_hidden_states = current_ip_hidden_states.reshape(
                        batch_size, -1, attn.heads * attn.head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)
                    ip_attn_output += scale * current_ip_hidden_states

                return hidden_states, encoder_hidden_states, ip_attn_output
            else:
                return hidden_states, encoder_hidden_states
        else:
            return main_processor(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)



class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = OptimizedFluxAttnProcessor
    _available_processors = [
        OptimizedFluxAttnProcessor,
        OptimizedFluxIPAdapterAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
        use_flash_attn: bool = True,
        enable_kv_cache: bool = False,
        fused_projections: bool = False,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.fused_projections = fused_projections

        # Normalization layers
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        # QKV projections - can be fused for better performance
        if fused_projections:
            self.to_qkv = torch.nn.Linear(query_dim, self.inner_dim * 3, bias=bias)
        else:
            self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
            self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
            self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # Output projection
        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(torch.nn.Dropout(dropout))

        # Cross-attention projections
        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            
            if fused_projections:
                self.to_added_qkv = torch.nn.Linear(added_kv_proj_dim, self.inner_dim * 3, bias=added_proj_bias)
            else:
                self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
                self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
                self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        # Set processor
        if processor is None:
            processor = self._default_processor_cls(
                use_flash_attn=use_flash_attn,
                enable_kv_cache=enable_kv_cache
            )
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0,
                 use_flash_attn: bool = True, enable_grad_checkpointing: bool = False):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.enable_grad_checkpointing = enable_grad_checkpointing

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=OptimizedFluxAttnProcessor(use_flash_attn=use_flash_attn),
            eps=1e-6,
            pre_only=True,
            use_flash_attn=use_flash_attn
        )

    def _forward_impl(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs):
        """Implementation of forward pass"""
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs,
                use_reentrant=False
            )
        else:
            return self._forward_impl(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)


@maybe_allow_in_graph  
class FluxTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6,
        use_flash_attn: bool = True, enable_grad_checkpointing: bool = False
    ):
        super().__init__()
        self.enable_grad_checkpointing = enable_grad_checkpointing

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=OptimizedFluxAttnProcessor(use_flash_attn=use_flash_attn),
            eps=eps,
            use_flash_attn=use_flash_attn
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _forward_impl(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs,
                use_reentrant=False
            )
        else:
            return self._forward_impl(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)


class FluxPosEmbed(nn.Module):    
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.cache = {}  
        self.cache_lock = threading.RLock()
        
    def _compute_embeddings(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (tuple(ids.shape), ids.device.type, str(ids.dtype))
        
        with self.cache_lock:
            if cache_key in self.cache:
                cached_cos, cached_sin = self.cache[cache_key]
                if (cached_cos.device == ids.device and 
                    cached_cos.dtype == ids.dtype and
                    cached_cos.shape[0] >= ids.shape[0]):
                    return cached_cos[:ids.shape[0]], cached_sin[:ids.shape[0]]

        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        
        # Cache result
        with self.cache_lock:
            self.cache[cache_key] = (freqs_cos, freqs_sin)
            # Limit cache size
            if len(self.cache) > 20:
                # Remove oldest entries
                keys_to_remove = list(self.cache.keys())[:-15]
                for key in keys_to_remove:
                    del self.cache[key]
        
        return freqs_cos, freqs_sin

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._compute_embeddings(ids)


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        use_flash_attn: bool = True,
        enable_embedding_cache: bool = True,
        embedding_cache_size: int = 200,
        enable_memory_efficient_attention: bool = True,
        enable_fused_projections: bool = False,
        enable_optimized_grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.use_flash_attn = use_flash_attn
        self.enable_embedding_cache = enable_embedding_cache
        self.enable_memory_efficient_attention = enable_memory_efficient_attention

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        # Context and input embedders
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                use_flash_attn=use_flash_attn,
                enable_grad_checkpointing=enable_optimized_grad_checkpointing,
            )
            for _ in range(num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                use_flash_attn=use_flash_attn,
                enable_grad_checkpointing=enable_optimized_grad_checkpointing,
            )
            for _ in range(num_single_layers)
        ])

        # Output layers
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        if self.enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(max_size=embedding_cache_size)
        
        # Legacy gradient checkpointing flag
        self.gradient_checkpointing = False

    def _get_cached_embeddings(self, ids: torch.Tensor, timestep: Optional[torch.Tensor] = None,
                              guidance: Optional[torch.Tensor] = None, 
                              pooled_projections: Optional[torch.Tensor] = None) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enable_embedding_cache:
            return None
        
        cache_key = EmbeddingCacheKey.from_inputs(ids, timestep, guidance, pooled_projections)
        cached = self.embedding_cache.get(cache_key)
        
        if cached is not None:
            target_device = ids.device
            target_dtype = timestep.dtype if timestep is not None else torch.float32
            
            return {
                'image_rotary_emb': cached.to_device(target_device, target_dtype),
                'temb': cached.to_device(target_device, target_dtype) if hasattr(cached, 'temb') else None
            }
        
        return None

    def _cache_embeddings(self, cache_key: EmbeddingCacheKey, image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
                         temb: torch.Tensor):
        if not self.enable_embedding_cache:
            return
            
        # Convert to cacheable format
        cached_embedding = CachedEmbedding(
            embedding=image_rotary_emb,
            device=str(image_rotary_emb[0].device),
            dtype=str(image_rotary_emb[0].dtype)
        )
        
        self.embedding_cache.put(cache_key, cached_embedding)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `joint_attention_kwargs` when not using PEFT backend is ineffective.")

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        ids = torch.cat((txt_ids, img_ids), dim=0)
        
        cache_key = EmbeddingCacheKey.from_inputs(ids, timestep, guidance, pooled_projections)
        cached_embeddings = self._get_cached_embeddings(ids, timestep, guidance, pooled_projections)
        
        if cached_embeddings:
            image_rotary_emb = cached_embeddings['image_rotary_emb']
            if cached_embeddings['temb']:
                temb = cached_embeddings['temb']
            else:
                temb = (
                    self.time_text_embed(timestep, pooled_projections)
                    if guidance is None
                    else self.time_text_embed(timestep, guidance, pooled_projections)
                )
        else:
            image_rotary_emb = self.pos_embed(ids)
            temb = (
                self.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.time_text_embed(timestep, guidance, pooled_projections)
            )
            
            # Cache the results
            self._cache_embeddings(cache_key, image_rotary_emb, temb)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning("Passing `txt_ids` 3d torch.Tensor is deprecated.")
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning("Passing `img_ids` 3d torch.Tensor is deprecated.")
            img_ids = img_ids[0]

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        if hasattr(self, 'embedding_cache'):
            return {
                'embedding_cache': self.embedding_cache.get_stats()
            }
        return {}

    def clear_caches(self):
        if hasattr(self, 'embedding_cache'):
            self.embedding_cache.clear()

        if hasattr(self.pos_embed, 'cache'):
            with self.pos_embed.cache_lock:
                self.pos_embed.cache.clear()
    
    def enable_flash_attention(self, enabled: bool = True):
        self.use_flash_attn = enabled and FLASH_ATTENTION_AVAILABLE
        
        def update_processor(module):
            if hasattr(module, 'processor') and hasattr(module.processor, 'use_flash_attn'):
                module.processor.use_flash_attn = self.use_flash_attn
        
        self.apply(update_processor)
    
    def set_memory_efficient_attention(self, enabled: bool = True):
        self.enable_memory_efficient_attention = enabled
        
        for module in self.modules():
            if hasattr(module, 'enable_memory_efficient_attention'):
                module.enable_memory_efficient_attention = enabled
    
    def get_memory_stats(self) -> Dict[str, Any]:
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
            }
        
        return stats
    
    def optimize_for_inference(self):
        self.eval()

        self.enable_flash_attention(True)
        self.set_memory_efficient_attention(True)
        
    
    def optimize_for_training(self):
        self.train()
        
        self.set_memory_efficient_attention(True)
        
        self.gradient_checkpointing = True
        
        self.clear_caches()