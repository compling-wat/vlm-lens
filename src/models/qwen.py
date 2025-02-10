"""qwen.py.

File for providing the Qwen model implementation.
"""
from typing import Any, Dict

from transformers import (AutoImageProcessor, AutoTokenizer,
                          Qwen2VLForConditionalGeneration)

from .base import ModelBase, ModelSelection
from .config import Config


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, model_path: str, config: Config):
        """Initialization of the qwen model.

        This makes sure to set the model name, path and config.

        Args:
            model_path (str): The path to the specific model
            config (Config): Parsed config
        """
        self.model_name = ModelSelection.QWEN
        self.model_path = model_path
        self.config = config

        # initialize the parent class
        super().__init__()

    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def classify_input_ids(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method to be implemented by each subclass.

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
            input_ids['image_input_ids'] = image_inputs['pixel_values']
        if inputs['txt']:
            text_inputs = self.tokenizer(inputs['txt'], return_tensors='pt')
            input_ids['text_input_ids'] = text_inputs['input_ids']

        return input_ids
