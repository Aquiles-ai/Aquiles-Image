import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # It's supposed to go faster

from platformdirs import user_data_dir
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Literal, Union
from pathlib import Path

## Constant

AQUILES_VIDEO_BASE_PATH = user_data_dir("aquiles_video", "Aquiles-Image")

os.makedirs(AQUILES_VIDEO_BASE_PATH, exist_ok=True)

BASE_WAN_2_2 = "lightx2v/Wan2.2-Official-Models"

BASE_WAN_2_2_FILE = "wan2.2_ti2v_lightx2v.safetensors"

ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"

REPO_ID_WAN_2_2_DISTILL = "lightx2v/Wan2.2-Distill-Models"

REPO_ID_WAN_2_2_LI = "lightx2v/Wan2.2-Lightning"

BASE_HY_1_5 = "tencent/HunyuanVideo-1.5"

def download_tokenizers():
    print(hf_hub_download(repo_id="lightx2v/Encoders", filename="special_tokens_map.json", subfolder="google/umt5-xxl", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2"))
    print(hf_hub_download(repo_id="lightx2v/Encoders", filename="spiece.model", subfolder="google/umt5-xxl", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2"))
    print(hf_hub_download(repo_id="lightx2v/Encoders", filename="tokenizer.json", subfolder="google/umt5-xxl", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2"))
    print(hf_hub_download(repo_id="lightx2v/Encoders", filename="tokenizer_config.json", subfolder="google/umt5-xxl", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2"))

def download_base_wan_2_2():
    print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan_2_2")
    print(snapshot_download(repo_id="Wan-AI/Wan2.2-T2V-A14B", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2"))

def download_wan_2_2_turbo():
    print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan_2_2_turbo")
    print(snapshot_download(repo_id="Aquiles-ai/Wan2.2-Turbo", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2_turbo"))

def download_hy(name: Literal["hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p", "hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8", "hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"] = "hunyuanVideo-1.5-480p"):
    if name == "hunyuanVideo-1.5-480p":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p")
        print(snapshot_download(repo_id="Aquiles-ai/HunyuanVideo-1.5-480p", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p"))
    elif name == "hunyuanVideo-1.5-720p":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/hy_1_5_720p")
        print(snapshot_download(repo_id="Aquiles-ai/HunyuanVideo-1.5-720p", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_720p"))
    elif name == "hunyuanVideo-1.5-480p-fp8":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_fp8")
        print(snapshot_download(repo_id="Aquiles-ai/HunyuanVideo-1.5-480p-fp8", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_fp8"))
    elif name == "hunyuanVideo-1.5-720p-fp8":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/hy_1_5_720p_fp8")
        print(snapshot_download(repo_id="Aquiles-ai/HunyuanVideo-1.5-720p-fp8", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_720p_fp8"))
    elif name == "hunyuanVideo-1.5-480p-turbo":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_turbo")
        print(snapshot_download(repo_id="Aquiles-ai/HunyuanVideo-1.5-480p-Turbo", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_turbo"))
    elif name == "hunyuanVideo-1.5-480p-turbo-fp8":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_turbo_fp8")
        print(snapshot_download(repo_id="Aquiles-ai/HunyuanVideo-1.5-480p-Turbo-fp8", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_turbo_fp8"))
    else:
        raise ValueError("Model not available")

def download_wan2_1(name: Literal["wan2.1", "wan2.1-turbo", "wan2.1-turbo-fp8", "wan2.1-3B"] = "wan2.1"):
    if name == "wan2.1":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan2_1")
        print(snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-14B", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan2_1"))
    elif name == "wan2.1-turbo":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan2_1_turbo")
        print(snapshot_download(repo_id="Aquiles-ai/Wan2.1-Turbo", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan2_1_turbo"))
    elif name == "wan2.1-turbo-fp8":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan2_1_turbo_fp8")
        print(snapshot_download(repo_id="Aquiles-ai/Wan2.1-Turbo-fp8", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan2_1_turbo_fp8"))
    elif name == "wan2.1-3B":
        print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan2_1_3b")
        print(snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan2_1_3b"))
    else:
        raise ValueError("Model not available")

def get_path_file_video_model(name: Literal["wan2.2", "wan2.2-turbo", "wan2.1", "wan2.1-turbo", "wan2.1-turbo-fp8", "wan2.1-3B", "hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p", "hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8", "hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"] = "wan2.2"):
    if name == "wan2.2":
        return f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2"
    elif name == "wan2.2-turbo":
        return f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2_turbo"
    elif name == "hunyuanVideo-1.5-480p":
        return f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p"
    elif name == "hunyuanVideo-1.5-720p":
        return f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_720p"  
    elif name == "hunyuanVideo-1.5-480p-fp8":
        return f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_fp8"
    elif name == "hunyuanVideo-1.5-720p-fp8":
        return f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_720p_fp8"
    elif name == "hunyuanVideo-1.5-480p-turbo":
        return f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_turbo"
    elif name == "hunyuanVideo-1.5-480p-turbo-fp8":
        return f"{AQUILES_VIDEO_BASE_PATH}/hy_1_5_480p_turbo_fp8"
    elif name == "wan2.1":
        return f"{AQUILES_VIDEO_BASE_PATH}/wan2_1"
    elif name == "wan2.1-turbo":
        return f"{AQUILES_VIDEO_BASE_PATH}/wan2_1_turbo"
    elif name == "wan2.1-turbo-fp8":
        return f"{AQUILES_VIDEO_BASE_PATH}/wan2_1_turbo_fp8"
    elif name == "wan2.1-3B":
        return f"{AQUILES_VIDEO_BASE_PATH}/wan2_1_3b"
    else:
        return None

def get_path_save_video(id_video: str):
    return f"{AQUILES_VIDEO_BASE_PATH}/results/{id_video}.mp4"

def file_exists(path: Union[str, Path, None]) -> bool:
    if path is None:
        return False

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p = p.resolve(strict=False)

    try:
        return p.is_file() and p.stat().st_size > 0
    except (OSError, FileNotFoundError):
        return False
