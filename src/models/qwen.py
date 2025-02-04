"""qwen.py.

File for providing the Qwen model implementation.
"""
from transformers import Qwen2VLForConditionalGeneration

from .base import ModelBase
from .config import Config


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, config: Config):
        """Initialization of the qwen model.

        Args:
            config (Config): Parsed config
        """
        self.IMG_LM_DIM = 129  # TODO: is there any way to automate this?

        # initialize the parent class
        super().__init__(config)

    def _register_subclass_hook(self, hook_fn):
        """Registers the hook_fn.

        Args:
            hook_fn (hook fn): The hook function to register
        """
        self.model.visual.blocks[-1].register_forward_hook(hook_fn)

    def _is_input_image(self, input):
        """Function that returns whether this input is an image embedding.

        Args:
            input (tensor): The input tensor provided.

        Returns:
            bool: Boolean flag
        """
        return input[0].shape[1] == self.IMG_LM_DIM

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )
