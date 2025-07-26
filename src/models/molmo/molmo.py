"""molmo.py.

File for providing the Molmo model implementation.
"""
import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from src.models.base import ModelBase
from src.models.config import Config


class MolmoModel(ModelBase):
    """Molmo model implementation."""

    def __init__(self, config: Config):
        """Initialization of the molmo model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **self.config.model,
            trust_remote_code=True
        ) if hasattr(self.config, 'model') else (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        )

    def _init_processor(self) -> None:
        """Initializes the processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def _generate_prompt(self) -> str:
        """Generates the Molmo model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor argument to be input into the processor.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor arguments per image and prompt.
        """
        if img_path is None:
            raise ValueError('Molmo cannot have text-only generation.')

        # prepare the data inputs according to
        # https://huggingface.co/allenai/Molmo-7B-D-0924
        data_inputs = self.processor.process(
            images=[Image.open(img_path)],
            text=prompt
        )

        # move inputs to the correct device and make a batch of size 1
        return {
            k: v.to(self.config.device).unsqueeze(0)
            for k, v in data_inputs.items()
        }

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data: The given data tensor.
        """
        generation_config = self.config.forward
        with torch.no_grad():
            _ = self.model.generate_from_batch(
                data,
                GenerationConfig(**generation_config),
                tokenizer=self.processor.tokenizer
            )
        logging.debug('Completed forward pass...')
