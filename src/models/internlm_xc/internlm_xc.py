"""internlm.py.

File for providing the InternLM-XComposer model implementation.
"""
import logging

import torch
from transformers import AutoModel, AutoProcessor

from src.models.base import ModelBase
from src.models.config import Config


class InternLMXComposerModel(ModelBase):
    """InternLM model implementation."""

    def __init__(self, config: Config):
        """Initialization of the InternLM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        )

    def _init_processor(self):
        """Overridden function to instantiate the model's processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.tokenizer = self.processor

    def _generate_prompt(self, add_generation_prompt=True):
        """Overridden function to generate the prompt for the model."""
        return self.config.prompt

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Overridden function to generate the format the prompt for the processor."""
        logging.debug('Loading data...')

        # Manually format input as we do not need a processor
        inputs = {}

        # Text prompts are required for this model
        if not hasattr(self.config, 'prompt'):
            raise RuntimeError('No input prompt was provided for the InternLM-XC model')

        # If there are images, load them and add image token to prompt
        if self.config.has_images():
            inputs['query'] = f'<ImageHere>; {prompt}'
            inputs['image'] = [img_path]

        else:
            inputs['query'] = prompt

        return inputs

    def _forward(self, data: dict):
        """Overridden function to run the model forward pass."""
        device_type = str(self.config.device)
        with torch.autocast(device_type=device_type):
            _, _ = self.model.chat(self.processor, **data, **self.config.forward)
