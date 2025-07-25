"""aya_vision.py.

File for providing the AyaVision model implementation.
"""

from transformers import AutoModelForImageTextToText

from src.models.base import ModelBase
from src.models.config import Config

class AyaVisionModel(ModelBase):
    """AyaVision model implementation."""

    def __init__(self, config: Config):
        """Initialization of the AyaVision model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Load the AyaVision model with proper configuration."""

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **getattr(self.config, "model", {})
        )
