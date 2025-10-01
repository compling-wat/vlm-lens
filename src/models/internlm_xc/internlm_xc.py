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

    def __init__(self, config: Config) -> None:
        """Initialization of the InternLM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
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

    def _init_processor(self) -> None:
        """Overridden function to instantiate the model's processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.model.tokenizer = self.processor

    def _generate_prompt(self, prompt: str, add_generation_prompt: bool = True, has_images: bool = False) -> str:
        """Overridden function to generate the prompt for the model.

        Args:
            prompt (str): The input prompt to be processed.
            add_generation_prompt (bool): Whether to add a start token of a bot response.
            has_images (bool): Whether the model has images or not.

        Returns:
            str: The formatted prompt ready for model input.
        """
        return prompt

    def _generate_processor_output(self, prompt: str, img_path: str) -> dict:
        """Overridden function to generate the format the prompt for the processor.

        Args:
            prompt (str): The input prompt to be processed.
            img_path (str): The path to the image to be processed.

        Returns:
            dict: The formatted inputs for the processor.

        Raises:
            ValueError: If no prompt is provided when required.
        """
        logging.debug('Loading data...')

        # Manually format input as we do not need a processor
        inputs = {}

        # Text prompts are required for this model
        if not prompt:
            raise ValueError(
                'No input prompt was provided for the InternLM-XC model')

        # If there are images, load them and add image token to prompt
        if self.config.has_images():
            inputs['query'] = f'<ImageHere>; {prompt}'
            inputs['image'] = [img_path]
        else:
            inputs['query'] = prompt

        return inputs

    def _forward(self, data: dict) -> None:
        """Overridden function to run the model forward pass.

        Args:
            data (dict): The input data for the model.
        """
        device_type = str(self.config.device)
        logging.debug(f'DATA: {data}')
        with torch.autocast(device_type=device_type):
            _, _ = self.model.chat(
                self.processor, **data, **self.config.forward)
