"""llava.py.

File for providing the Llava model implementation.
"""
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from src.models.base import ModelBase
from src.models.config import Config


class LlavaNextModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            LlavaNextForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def _init_processor(self) -> None:
        """Overridden function to populate self.processor."""
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
