"""perception_lm.py.

File for providing the Perception-LM model implementation.
"""

from transformers import AutoModelForImageTextToText, AutoProcessor

from src.models.base import ModelBase
from src.models.config import Config


class PerceptionLMModel(ModelBase):
    """Perception-LM model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialize the Perception-LM model.

        Args:
            config (Config): Parsed config.
        """
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate Perception-LM model.

        Huggingface token is required.
        Replace <HUGGINGFACE_TOKEN> in configs/perception-lm-1b.yaml file with your own hugging face security token.
        Note: 'token' is a general Hugging Face Hub access token, not specific to Perception-LM.
        It enables loading private models or authenticated access.
        See: https://huggingface.co/docs/hub/en/security-tokens
        """
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **self.config.model
        )

    def _init_processor(self) -> None:
        """Initialize the Perception-LM processor.

        Huggingface token is required.
        Replace <HUGGINGFACE_TOKEN> in configs/perception-lm-1b.yaml file with your own hugging face security token.
        Note: 'token' is a general Hugging Face Hub access token, not specific to Perception-LM.
        It enables loading private models or authenticated access.
        See: https://huggingface.co/docs/hub/en/security-tokens
        """
        self.processor = AutoProcessor.from_pretrained(self.model_path, token=self.config.model['token'])
