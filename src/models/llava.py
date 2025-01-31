"""llava.py.

File for providing the Llava model implementation.
"""
from transformers import LlavaForConditionalGeneration

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
