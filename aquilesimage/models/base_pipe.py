from abc import ABC, abstractmethod

class BasePipeline(ABC):

    def __init__(self, **kwargs):
        self.pipeline = None

    @abstractmethod
    def start(self):
        """Loads the model and prepares the pipeline for inference."""

    @abstractmethod
    def optimization(self):
        """Applies the optimizations specific to each pipeline."""