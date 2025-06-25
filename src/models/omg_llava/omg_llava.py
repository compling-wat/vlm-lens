"""omg-llava.py.

File for providing the omg-llava model implementation.
"""
from src.models.base import ModelBase
from src.models.config import Config


class OmgLlavaModel(ModelBase):
    """OMG-LLaVA model implementation."""

    def __init__(self, config: Config):
        """Initialization of the OMG-LLaVA model.

        Args:
            config (Config): Parsed config
        """
        super().__init__(config)

    def _load_specific_model(self):
        """Load base LLaVA model and apply OMG-LLaVA LoRA weights."""
