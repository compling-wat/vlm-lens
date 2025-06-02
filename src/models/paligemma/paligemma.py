"""paligemma.py.

File for providing the Paligemma model implementation.
"""
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


class PeligemmaModel(ModelBase):
    """PeligemmaModel model implementation."""

    def __init__(self, config: Config):
        """Initialization of the palegemma model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate Paligemma model. Huggingface token is required to get access to the model."""
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        )

    def _init_processor(self) -> None:
        """Initialize the InternVL processor. Huggingface token is required."""
        self.processor = AutoProcessor.from_pretrained(self.model_path, token=self.config.model['token'])

    def _generate_prompt(self) -> str:
        """Generates the Paligemma model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt
