"""pixtral.py.

File for providing the Pixtral model implementation.
"""
from transformers import AutoProcessor, LlavaForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


class PixtralModel(ModelBase):
    """Pixtral model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the Pixtral model.

        Args:
            config (Config): Parsed config
        """
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        kwargs = getattr(self.config, 'model', {}) or {}
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **kwargs
        )

    def _init_processor(self) -> None:
        """Overridden function to populate self.processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)
