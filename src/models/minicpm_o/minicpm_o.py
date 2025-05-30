"""minicpm_o.py.

File for providing the MiniCPM-o model implementation.
"""

import logging
import torch
from PIL import Image
from src.models.base import ModelBase
from src.models.config import Config
from transformers import AutoModel, AutoTokenizer

class MiniCPMOModel(ModelBase):
    """MiniCPM-o model implementation."""

    def __init__(self, config: Config):
        """Initialization of the MiniCPM-o model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModel.from_pretrained(
            self.model_path, **getattr(self.config, 'model', {})
        )

    def _generate_prompt(self) -> str:
        """Generates the MiniCPM-o model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _init_processor(self) -> None:
        """Initialize the MiniCPM-o Tokenizer"""
        self.processor = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        image = Image.open(img_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [prompt, image]}]
        return {'msgs': msgs}

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        with torch.no_grad():
            _ = self.model.chat(**data, context=None, tokenizer=self.tokenizer, max_new_tokens=1)
        logging.debug('Completed forward pass...')