"""qwen.py.

File for providing the Qwen model implementation.
"""
from abc import ABC

from transformers import Qwen2VLForConditionalGeneration

from .base import ModelBase, ModelSelection
from .config import Config


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, model_path: str, config: Config):
        """Initialization of the qwen model.

        This makes sure to set the model name, path and config.

        Args:
            model_path (str): The path to the specific model
            config (Config): Parsed config
        """
        self.model_name = ModelSelection.QWEN
        self.model_path = model_path
        self.config = config
        self.IMG_LM_DIM = 129  # TODO: is there any way to automate this?

        # initialize the parent class
        super().__init__()

    def register_subclass_hook(self, vis, hook_fn):
        """Registers the hook_fn based on whether it's a vision only embedding.

        Args:
            vis (bool): Determines whether it's a vision only
            hook_fn (hook fn): The hook function to register
        """
        if vis:
            self.model.visual.blocks[-1].register_forward_hook(hook_fn)
        else:
            self.model.lm_head.register_forward_hook(hook_fn)

    def is_input_image(self, input):
        """Function that returns whether this input is an image embedding.

        Args:
            input (tensor): The input tensor provided.

        Returns:
            bool: Boolean flag
        """
        return input[0].shape[1] == self.IMG_LM_DIM



    def load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )
