from platformdirs import user_data_dir
import os

## Constant

AQUILES_VIDEO_BASE_PATH = user_data_dir("aquiles_video", "Aquiles-Image")

os.makedirs(AQUILES_VIDEO_BASE_PATH, exist_ok=True)

BASE_WAN_2_2 = "lightx2v/Wan2.2-Official-Models"

REPO_ID_WAN_2_2_DISTILL = "lightx2v/Wan2.2-Distill-Models"

REPO_ID_WAN_2_2_LI = "lightx2v/Wan2.2-Lightning"

BASE_HY_1_5 = "tencent/HunyuanVideo-1.5"