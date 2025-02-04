"""llava.py.

File for providing the Llava model implementation.
"""
from transformers import LlavaForConditionalGeneration

from .base import ModelBase
from .config import Config


class LlavaModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, config: Config):
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        self.IMG_LM_DIM = 599  # TODO: is there any way to automate this?

        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            LlavaForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def _register_subclass_hook(self, hook_fn):
        """Registers the hook_fn.

        Args:
            hook_fn (hook fn): The hook function to register
        """
        self.model.vision_tower.vision_model.\
            encoder.layers[-1].register_forward_hook(hook_fn)

    def _is_input_image(self, input):
        """Function that returns whether this input is an image embedding.

        Args:
            input (tensor): The input tensor provided.

        Returns:
            bool: Boolean flag
        """
        return input[0].shape[1] == self.IMG_LM_DIM
