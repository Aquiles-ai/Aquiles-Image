from platformdirs import user_data_dir
import json
import aiofiles
import asyncio
from pathlib import Path
import os
from aquilesimage.models import ConfigsServe
from typing import Dict, Any
import time
import threading
from typing import Optional

_load_lock = asyncio.Lock()
data_dir = user_data_dir("aquiles", "Aquiles-Image")
os.makedirs(data_dir, exist_ok=True)
AQUILES_CONFIG = os.path.join(data_dir, "aquiles_image_config.json")
_cache_lock = threading.Lock()
_cached_config: Optional[Dict[str, Any]] = None
_cache_timestamp: float = 0
_cache_mtime: float = 0


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