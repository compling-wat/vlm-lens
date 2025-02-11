"""llava.py.

File for providing the Llava model implementation.
"""
from typing import Any, Dict

from transformers import (AutoImageProcessor, AutoTokenizer,
                          LlavaForConditionalGeneration)

from .base import ModelBase, ModelSelection
from .config import Config


class LlavaModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, model_path: str, config: Config):
        """Initialization of the llava model.

        This makes sure to set the model name, path and config.

        Args:
            model_path (str): The path to the specific model
            config (Config): Parsed config
        """
        self.model_name = ModelSelection.LLAVA
        self.model_path = model_path
        self.config = config

        # initialize the parent class
        super().__init__()

    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            LlavaForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def classify_input_ids(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Separate image and text inputs and convert them to input ids.

        Args:
            inputs          : image and text inputs.

        Returns:
            input_ids       : image and text input ids.
        """
        # load models
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # decode input ids
        input_ids = {}
        if inputs['img']:
            image_inputs = self.image_processor(inputs['img'], return_tensors='pt')
            input_ids['img'] = image_inputs['pixel_values']
        if inputs['txt']:
            text_inputs = self.tokenizer(inputs['txt'], return_tensors='pt')
            input_ids['txt'] = text_inputs['input_ids']

        return input_ids
