"""paligemma.py.

File for providing the Paligemma model implementation.
"""
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


class PaligemmaModel(ModelBase):
    """PaligemmaModel model implementation."""

    def __init__(self, config: Config):
        """Initialization of the paligemma model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate Paligemma model.

        Huggingface token is required to get access to the model.
        Note: 'token' is a general Hugging Face Hub access token, not specific to PaliGemma.
        It enables loading private models or authenticated access.
        See: https://huggingface.co/docs/hub/en/security-tokens
        """
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        )

    def _init_processor(self) -> None:
        """Initialize the Paligemma processor.

        Huggingface token is required.
        Note: 'token' is a general Hugging Face Hub access token, not specific to PaliGemma.
        It enables loading private models or authenticated access.
        See: https://huggingface.co/docs/hub/en/security-tokens
        """
        self.processor = AutoProcessor.from_pretrained(self.model_path, token=self.config.model['token'])

    def _generate_prompt(self) -> str:
        """Generates the Paligemma model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt
