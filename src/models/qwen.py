"""qwen.py.

File for providing the Qwen model implementation.
"""
from transformers import Qwen2VLForConditionalGeneration

from .base import ModelBase, ModelSelection


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, model_path: str):
        """Initialization of the qwen model.

        This makes sure to set the model name and path.
        """
        self.model_name = ModelSelection.QWEN
        self.model_path = model_path
        super().__init__()

    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path
        )
