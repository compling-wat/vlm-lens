"""minicpm.py.

File for providing the MiniCPM model implementation.
"""

import logging

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from src.models.base import ModelBase
from src.models.config import Config


class MiniCPMModel(ModelBase):
    """MiniCPM model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the MiniCPM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = AutoModel.from_pretrained(
            self.model_path, **getattr(self.config, 'model', {})
        )

    def _generate_prompt(self, prompt: str) -> str:
        """Generates the MiniCPM model prompt which will not use the chat template.

        Args:
            prompt (str): The prompt content.

        Returns:
            str: The prompt to return, set by the config.
        """
        return prompt

    def _init_processor(self) -> None:
        """Initialize the MiniCPM tokenizer."""
        self.processor = None  # no intended processor here
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _generate_processor_output(self, prompt: str, img_path: str) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        msgs = [{'role': 'user', 'content': prompt}]
        image = Image.open(img_path).convert('RGB')
        return {'msgs': msgs, 'image': image}

    def _forward(self, data: BatchFeature) -> None:
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        with torch.no_grad():
            _ = self.model.chat(**data, context=None, tokenizer=self.tokenizer, **self.config.forward)
        logging.debug('Completed forward pass...')
