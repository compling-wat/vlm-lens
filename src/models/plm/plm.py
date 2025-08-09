"""plm.py.

File for providing the Plm model implementation.
"""

import logging

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from src.models.base import ModelBase
from src.models.config import Config


class PlmModel(ModelBase):
    """PLM model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the PLM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModelForImageTextToText.from_pretrained(
                self.model_path
            )
        )

    def _init_processor(self) -> None:
        """Initialize the self.processor by loading from the path."""
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)

    def _generate_prompt(self, prompt: str) -> str:
        """Generates the PLM model prompt which will not use the chat template.

        Args:
            prompt (str): The input prompt to be processed.

        Returns:
            str: The prompt to return, set by the config.
        """
        return f'USER: <image>\n{prompt} ASSISTANT:'  # Taken directly from official doc

    def _forward(self, data: BatchFeature) -> None:
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        with torch.no_grad():
            _ = self.model.generate(**data, **self.config.forward)
        logging.debug('Completed forward pass...')
