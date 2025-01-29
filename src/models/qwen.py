"""qwen.py.

File for providing the Qwen model implementation.
"""
from transformers import Qwen2VLForConditionalGeneration

from .base import ModelBase, ModelSelection


class QwenModel(ModelBase):
    """Qwen 7B model implementation."""

    def __init__(self):
        """Initialization of the qwen model.

        This makes sure to set the model name and path.
        """
        self.model_name = ModelSelection.QWEN
        self.model_path = 'Qwen/Qwen2-VL-7B-Instruct'
        super().__init__()

    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = \
            Qwen2VLForConditionalGeneration.from_pretrained(self.model_path)


class QwenModel_2B(ModelBase):
    """Qwen 2B model implementation."""

    def __init__(self):
        """Initialization of the qwen model.

        This makes sure to set the model name and path.
        """
        self.model_name = ModelSelection.QWEN_2B
        self.model_path = 'Qwen/Qwen2-VL-2B-Instruct'
        super().__init__()

    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = \
            Qwen2VLForConditionalGeneration.from_pretrained(self.model_path)
