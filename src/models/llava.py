"""llava.py.

File for providing the Llava model implementation.
"""
from transformers import LlavaForConditionalGeneration

from .base import ModelBase, ModelSelection


class LlavaModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, model_path: str):
        """Initialization of the llava model.

        This makes sure to set the model name and path.
        """
        self.model_name = ModelSelection.LLAVA
        self.model_path = model_path

        # initialize the parent class
        super().__init__()

    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path
        )
