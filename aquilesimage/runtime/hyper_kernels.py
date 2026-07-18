from aquilesimage.utils import setup_colored_logger
from aquilesimage.models import BatchCompile
import logging
from typing import Any, List

logger_p = setup_colored_logger("Aquiles-Image-Runtime", logging.DEBUG)

class HyperKernels:
    def __init__(self, pipeline: Any, b_to_compile: List[BatchCompile]):
        self.pipeline = pipeline
        self.b_to_compile = b_to_compile

        n_shapes = len(b_to_compile)
        batch_sizes = sorted({item.b for item in b_to_compile})
        resolutions = sorted({(item.h, item.w) for item in b_to_compile})

        logger_p.info(f"HyperKernels initialized")
        logger_p.info(f"  Shapes to compile: {n_shapes}")
        logger_p.info(f"  Batches: {batch_sizes}")
        logger_p.info(f"  Resolutions: {resolutions}")

    def compiles(self):
        logger_p.info("")
        batch_sizes = sorted({item.b for item in self.b_to_compile})
        resolutions = sorted({(item.h, item.w) for item in self.b_to_compile})
        logger_p.info("Starting warmup compilation")
        logger_p.info(f" batches={batch_sizes} ")
        logger_p.info(f" resolutions={resolutions} ")
        try:
            self.pipeline.warmup_compile(
                batch_sizes=batch_sizes,
                resolutions=resolutions,
            )
        except Exception as e:
            logger_p.error(f"Failed to run warmup_compile: {e}")