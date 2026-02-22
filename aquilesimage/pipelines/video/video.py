from aquilesimage.pipelines.video.hy import HunyuanVideo_Pipeline
from aquilesimage.pipelines.video.ltx_2 import LTX_2_Pipeline
from aquilesimage.pipelines.video.wan import Wan2_1_Pipeline, Wan2_2_Pipeline, Wan2_2_Turbo_Pipeline

class ModelVideoPipelineInit:
    def __init__(self, model: str):
        self.model = model
        self.pipeline = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        if self.model == 'wan2.2':
            self.pipeline = Wan2_2_Pipeline()

        elif self.model == 'wan2.2-turbo':
            self.pipeline = Wan2_2_Turbo_Pipeline()
        
        elif self.model in ["hunyuanVideo-1.5-480p", "hunyuanVideo-1.5-720p", "hunyuanVideo-1.5-480p-fp8", "hunyuanVideo-1.5-720p-fp8", "hunyuanVideo-1.5-480p-turbo", "hunyuanVideo-1.5-480p-turbo-fp8"]:
            self.pipeline = HunyuanVideo_Pipeline(self.model)

        elif self.model in ["wan2.1", "wan2.1-3B", "wan2.1-turbo", "wan2.1-turbo-fp8"]:
            self.pipeline = Wan2_1_Pipeline(self.model)
        
        elif self.model == "ltx-2":
            self.pipeline = LTX_2_Pipeline(self.model)

        return self.pipeline