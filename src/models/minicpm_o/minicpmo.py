"""minicpmo.py.

File for providing the MiniCPM-o model implementation.
"""

import logging

from src.models.base import ModelBase
from src.models.config import Config


class MiniCPMModel(ModelBase):

    def __init__(self, config: Config):
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        pass

    def _generate_prompt(self) -> str:
        pass

    def _init_processor(self) -> None:
        pass

    def _generate_processor_output(self, prompt, img_path) -> dict:
        pass

    def _forward(self, data: BatchFeature):
        pass