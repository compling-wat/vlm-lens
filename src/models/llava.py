"""llava.py.

File for providing the Llava model implementation.
"""
from transformers import LlavaForConditionalGeneration

from .base import ModelBase
from .config import Config


class LlavaModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, config: Config):
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            LlavaForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )
    def run(self):
        self.forward(self.load_input_data())
        self.save_states()
