"""clip.py.

File for providing the Clip model implementation.
"""

from transformers import CLIPModel

from src.models.base import ModelBase
from src.models.config import Config


class ClipModel(ModelBase):
    """Clip model implementation."""

    def __init__(self, config: Config):
        """Initialization of the clip model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = CLIPModel.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            CLIPModel.from_pretrained(
                self.model_path
            )
        )

    def _generate_prompt(self) -> str:
        """Generates the CLIP model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt
