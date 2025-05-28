"""internvl.py.

File for providing the Intern-VL model implementation.
"""

import logging

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from src.models.base import ModelBase
from src.models.config import Config
from src.models.internvl.utils import load_image


class InternVLModel(ModelBase):
    """InternVL model implementation."""

    def __init__(self, config: Config):
        """Initialization of the InternVL model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModel.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModel.from_pretrained(
                self.model_path
            )
        )

    def _generate_prompt(self) -> str:
        """Generates the InternVL model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _init_processor(self) -> None:
        """Initialize the InternVL processor which need to be done manually."""
        self.processor = None  # no intended processor here
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.img_processor = load_image

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        return {'prompt': prompt, 
                'pixel_values': None if img_path is None else self.img_processor(img_path, max_num=12).to(dtype=torch.bfloat16).to(self.config.device)}

    def _forward(self, data: BatchFeature):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        generation_config = dict(max_new_tokens=1, do_sample=True)
        with torch.no_grad():
            _ = self.model.chat(self.tokenizer, data['pixel_values'], data['prompt'], generation_config)
        logging.debug('Completed forward pass...')
