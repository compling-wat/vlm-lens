"""qwen.py.

File for providing the Qwen model implementation.
"""
from transformers import Qwen2VLForConditionalGeneration

from .base import ModelBase
from .config import Config


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, config: Config):
        """Initialization of the qwen model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def run(self):
        """Run the model and save output states."""
        self.forward(self.load_input_data())
        self.save_states()
