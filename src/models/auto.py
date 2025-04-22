"""auto.py.

File for providing model implementations for any models using AutoModel.
"""

from transformers import AutoModelForVision2Seq

from .base import ModelBase
from .config import Config


class AutoModelBase(ModelBase):
    """Auto model implementation."""

    def __init__(self, config: Config):
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModelForVision2Seq.from_pretrained(
                self.model_path
            )
        )
