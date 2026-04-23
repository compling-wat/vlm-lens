"""paligemma.py.

File for providing the Paligemma model implementation.
"""
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from src.models.base import ModelBase
from src.models.config import Config


def _strip_placeholder_token(model_cfg: dict) -> dict:
    """Return a copy of ``model_cfg`` without the example-YAML placeholder
    token. A bare ``<HUGGINGFACE_TOKEN>`` string is not a real credential — if
    left in place, ``from_pretrained`` authenticates with the literal string
    and returns 401 on gated repos. Dropping the key lets HF fall back to its
    standard token resolution (``HF_TOKEN`` env var, ``huggingface-cli login``
    cache).
    """
    if not isinstance(model_cfg, dict):
        return {}
    out = dict(model_cfg)
    tok = out.get('token')
    if isinstance(tok, str) and tok.startswith('<') and tok.endswith('>'):
        out.pop('token')
    return out


class PaligemmaModel(ModelBase):
    """PaligemmaModel model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the paligemma model.

        Args:
            config (Config): Parsed config
        """
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate Paligemma model.

        A HuggingFace access token with read permission to the gated PaliGemma
        repo is required. Supply it either as ``token: <your-token>`` under
        ``model:`` in the YAML, or via the standard ``HF_TOKEN`` environment
        variable.
        """
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path, **_strip_placeholder_token(self.config.model)
        )

    def _init_processor(self) -> None:
        """Initialize the Paligemma processor. See ``_load_specific_model`` for
        token-resolution rules."""
        kwargs = _strip_placeholder_token(self.config.model)
        token = kwargs.get('token')
        self.processor = (
            AutoProcessor.from_pretrained(self.model_path, token=token)
            if token else AutoProcessor.from_pretrained(self.model_path)
        )

    def _generate_prompt(self, prompt: str, add_generation_prompt: bool = True, has_images: bool = False) -> str:
        """Generates the Paligemma model prompt which will not use the chat template.

        Args:
            prompt (str): The input prompt for the model.
            add_generation_prompt (bool): Whether to add a start token of a bot response.
            has_images (bool): Whether the model has images or not.

        Returns:
            str: The prompt to return, set by the config.
        """
        return prompt
