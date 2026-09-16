from platformdirs import user_data_dir
import hashlib
import json
import aiofiles
import asyncio
from pathlib import Path
import os
import platform
import sys
from aquilesimage.models import ConfigsServe, LoRAConfig
from typing import Dict, Any
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger("Aquiles-Image-Configs")

_load_lock = asyncio.Lock()
data_dir = user_data_dir("aquiles", "Aquiles-Image")
os.makedirs(data_dir, exist_ok=True)
AQUILES_CONFIG = os.path.join(data_dir, "aquiles_image_config.json")
_cache_lock = threading.Lock()
_cached_config: Optional[Dict[str, Any]] = None
_cache_timestamp: float = 0
_cache_mtime: float = 0
AQUILES_INDUCTOR_CACHE = f"{data_dir}/aquiles_inductor_cache"
os.makedirs(AQUILES_INDUCTOR_CACHE, exist_ok=True)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = AQUILES_INDUCTOR_CACHE
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

# HyperKernels cache-key version. Bump when the compile fingerprint changes
# (inductor opts, dynamic flag, fuse_qkv/channels_last, attention backend
# priority) so old dirs are never silently reused.
HYPERKERNELS_CACHE_KEY_VERSION = "hk-v1"

# Fingerprint of the compile options applied in piecewise mode
# (see pipelines/image/flux/flux_pipeline.py::optimization).
# Keep in sync with that function; any change here invalidates old caches.
HYPERKERNELS_COMPILE_FINGERPRINT: Dict[str, Any] = {
    "dynamic": False,
    "recompile_limit": 32,
    "conv_1x1_as_mm": True,
    "coordinate_descent_check_all_directions": False,
    "coordinate_descent_tuning": False,
    "epilogue_fusion": False,
    "shape_padding": True,
    "fuse_qkv": True,
    "channels_last": True,
}


def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version(name)
    except Exception:
        return None


def _safe_cuda_version() -> Optional[str]:
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


def _safe_cuda_arch() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no-cuda"
        try:
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm_{major}{minor}"
        except Exception:
            return "cuda-unknown-arch"
    except Exception:
        return "unknown"


def _safe_triton_version() -> Optional[str]:
    try:
        import triton

        return getattr(triton, "__version__", "installed")
    except Exception:
        return None


def build_hyperkernels_cache_components(
    model_name: Optional[str] = None,
    mode: str = "piecewise",
) -> Dict[str, Any]:
    return {
        "key_version": HYPERKERNELS_CACHE_KEY_VERSION,
        "model": model_name or "unknown-model",
        "mode": mode,
        "python": platform.python_version(),
        "torch": _safe_pkg_version("torch"),
        "cuda": _safe_cuda_version(),
        "cuda_arch": _safe_cuda_arch(),
        "triton": _safe_triton_version() or _safe_pkg_version("triton"),
        "diffusers": _safe_pkg_version("diffusers"),
        "transformers": _safe_pkg_version("transformers"),
        "compile": dict(HYPERKERNELS_COMPILE_FINGERPRINT),
    }


def build_hyperkernels_cache_key(
    model_name: Optional[str] = None,
    mode: str = "piecewise",
) -> str:
    components = build_hyperkernels_cache_components(model_name, mode)
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{HYPERKERNELS_CACHE_KEY_VERSION}-{digest}"


def get_versioned_inductor_cache_dir(
    model_name: Optional[str] = None,
    mode: str = "piecewise",
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    base = base_dir or AQUILES_INDUCTOR_CACHE
    key = build_hyperkernels_cache_key(model_name, mode)
    path = os.path.join(base, key)
    return {
        "base_dir": base,
        "key": key,
        "path": path,
        "triton_dir": os.path.join(path, "triton"),
        "components": build_hyperkernels_cache_components(model_name, mode),
    }


def _is_cache_warm(path: str) -> bool:
    try:
        if not os.path.isdir(path):
            return False
        with os.scandir(path) as it:
            for _ in it:
                return True
        return False
    except OSError:
        return False


def ensure_inductor_cache(
    model_name: Optional[str] = None,
    mode: str = "piecewise",
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = get_versioned_inductor_cache_dir(model_name, mode, base_dir)
    path = resolved["path"]
    triton_dir = resolved["triton_dir"]
    key = resolved["key"]

    warm = _is_cache_warm(path)

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logger.warning(f"Inductor cache dir not writable ({path}): {e}. Falling back to {AQUILES_INDUCTOR_CACHE}")
        path = AQUILES_INDUCTOR_CACHE
        triton_dir = os.path.join(path, "triton")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
        resolved["path"] = path
        resolved["triton_dir"] = triton_dir
        warm = _is_cache_warm(path)

    try:
        os.makedirs(triton_dir, exist_ok=True)
    except OSError as e:
        logger.warning(f"Triton cache dir not writable ({triton_dir}): {e}")

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TRITON_CACHE_DIR"] = triton_dir

    try:
        import torch._inductor.config as _inductor_config

        _inductor_config.cache_dir = path
    except Exception:
        pass

    comps = resolved["components"]
    logger.info(
        "HyperKernels cache %s: key=%s dir=%s (torch=%s cuda=%s arch=%s diffusers=%s model=%s)",
        "HIT" if warm else "MISS",
        key,
        path,
        comps.get("torch"),
        comps.get("cuda"),
        comps.get("cuda_arch"),
        comps.get("diffusers"),
        comps.get("model"),
    )

    resolved["hit"] = warm
    return resolved

def load_lora_config(path: str) -> LoRAConfig | None:
    try:
        resolved = Path(path).resolve()

        if not resolved.exists():
            logger.error(f"LoRA config file not found: {resolved}")
            return None

        if not resolved.is_file():
            logger.error(f"LoRA config path is not a file: {resolved}")
            return None

        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)

        return LoRAConfig(**data)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in LoRA config file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading LoRA config: {e}")
        return None

def config_file_exists() -> bool:
    return Path(AQUILES_CONFIG).exists()


def load_config_cli(use_cache: bool = True, cache_ttl: float = 30.0) -> Dict[str, Any]:
    global _cached_config, _cache_timestamp, _cache_mtime
    config_path = Path(AQUILES_CONFIG)
    if not config_path.exists():
        return {}
    current_time = time.time()
    if use_cache:
        with _cache_lock:
            try:
                file_mtime = config_path.stat().st_mtime
                
                if (_cached_config is not None and 
                    (current_time - _cache_timestamp) < cache_ttl and
                    file_mtime == _cache_mtime):
                    return _cached_config.copy()
                    
            except OSError:
                pass
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            
        if use_cache:
            with _cache_lock:
                try:
                    file_mtime = config_path.stat().st_mtime
                    _cached_config = config_data.copy()
                    _cache_timestamp = current_time
                    _cache_mtime = file_mtime
                except OSError:
                    pass
                    
        return config_data
        
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


async def load_config_app() -> Dict[str, Any]:
    async with _load_lock:  
        try:
            async with aiofiles.open(AQUILES_CONFIG, "r", encoding="utf-8") as f:
                s = await f.read()
        except FileNotFoundError:
            return {}
        except Exception as exc:
            return {}

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}

def clear_config_cache() -> None:
    global _cached_config, _cache_timestamp, _cache_mtime
    
    with _cache_lock:
        _cached_config = None
        _cache_timestamp = 0
        _cache_mtime = 0

def configs_image_serve(cfg: ConfigsServe, force: bool = False) -> None:
    conf = cfg.model_dump()
    config_path = Path(AQUILES_CONFIG)
    config_path.parent.mkdir(parents=True, exist_ok=True)


    if config_path.exists() and not force:
        try:
            existing_conf = load_config_cli()
            for key, value in conf.items():
                if value is not None:
                    existing_conf[key] = value
            conf = existing_conf
        except Exception:
            pass
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)
        
        clear_config_cache()
          
    except (OSError, UnicodeEncodeError) as e:
        raise Exception(f"Error saving configuration: {e}")


def create_basic_config_if_not_exists(model: str |  None = None, load_model: bool | None = None) -> bool:
    if config_file_exists():
        return False
    
    try:
        default_model = model or "stabilityai/stable-diffusion-3.5-medium"
        if load_model:
            basic_config = ConfigsServe(model=default_model, load_model=load_model)
        else:
            basic_config = ConfigsServe(model=default_model)
        configs_image_serve(basic_config, force=True)
        return True
    except Exception as e:
        raise Exception(f"Error creating basic configuration: {e}")

def get_inductor_cache_dir():
    return AQUILES_INDUCTOR_CACHE