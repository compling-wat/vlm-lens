"""auto.py.

File for providing model implementations for any models using AutoModel.
"""

from transformers import AutoModelForVision2Seq

from .base import ModelBase
from .config import Config


class GlammModel(ModelBase):
    """Glamm model implementation."""

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

    def _init_processor(self) -> None:
        """Don't set the self.processor as it's not needed for Glamm."""
        return

    def _generate_prompt(self) -> str:
        """Generates the GLaMM model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt
