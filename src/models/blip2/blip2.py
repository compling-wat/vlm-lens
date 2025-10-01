"""blip2.py.

File for providing the Blip2 model implementation.
"""

from transformers import Blip2ForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


class Blip2Model(ModelBase):
    """Blip-2 model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the Blip-2 model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Blip2ForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def _generate_prompt(self, prompt: str, add_generation_prompt: bool = True, has_images: bool = False) -> str:
        """Generates the BLIP-2 model prompt which will not use the chat template.

        Args:
            prompt (str): The input prompt to be processed.
            add_generation_prompt (bool): Whether to add a start token of a bot response.
            has_images (bool): Whether the model has images or not.

        Returns:
            str: The prompt to return, set by the config.
        """
        return prompt
