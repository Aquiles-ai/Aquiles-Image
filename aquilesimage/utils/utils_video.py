from platformdirs import user_data_dir
import os
from huggingface_hub import hf_hub_download

## Constant

AQUILES_VIDEO_BASE_PATH = user_data_dir("aquiles_video", "Aquiles-Image")

os.makedirs(AQUILES_VIDEO_BASE_PATH, exist_ok=True)

BASE_WAN_2_2 = "lightx2v/Wan2.2-Official-Models"

REPO_ID_WAN_2_2_DISTILL = "lightx2v/Wan2.2-Distill-Models"

REPO_ID_WAN_2_2_LI = "lightx2v/Wan2.2-Lightning"

BASE_HY_1_5 = "tencent/HunyuanVideo-1.5"

def download_base_wan_2_2():
    print(f"PATH: {AQUILES_VIDEO_BASE_PATH}/wan_2_2")
    hf_hub_download(repo_id=BASE_WAN_2_2, filename="wan2.2_ti2v_lightx2v.safetensors", local_dir=f"{AQUILES_VIDEO_BASE_PATH}/wan_2_2")