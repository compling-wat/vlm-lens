"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum

from transformers import AutoProcessor


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""
    LLAVA = 'llava'
    QWEN = 'qwen'


class ModelBase(ABC):
    """Provides an abstract base class for everything to implement."""

    def __init__(self):
        """Initialization of the model base class."""
        assert self.model_path is not None
        self.load_model()

    def load_model(self):
        """Loads the model and sets the processor from the loaded model."""
        logging.debug(
            f'Loading model {self.model_name.value}; {self.model_path}'
        )
        self.load_specific_model()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_state_hook(self, vis):
        """Generates the state hook depending on the embedding type.

        Args:
            vis (bool): Set to true if we want only image embeddings

        Returns:
            hook function: The hook function to return depending on the vis flag
        """
        def generate_vis_state_hook(module, input, output):
            """Hook handle function that returns only the image hidden states.

            This is to be used for the vision encoder.

            Args:
                module: The module
                input: The input
                output: The image embeddings
            """
            self.vis_image_states = output

        def generate_img_vis_state_hook(module, input, output):
            """Hook handle function that returns both image and text states.

            This hook here is to be used in the LM head.

            Args:
                module: The module
                input: The input
                output: The image embeddings
            """
            if self.is_input_image(input):
                self.vis_image_states = output
            else:
                self.txt_hidden_states = output

        return generate_vis_state_hook if vis else generate_img_vis_state_hook

    def register_hook(self, vis):
        """Registers the hook depending on the embedding setting.

        Args:
            vis (bool): Set to true in the image only embedding setting.
        """
        logging.debug('Generating hook function')
        self.register_subclass_hook(vis, self.generate_state_hook(vis))

    @abstractmethod
    def load_specific_model(self):
        """Abstract method that loads the specific model."""
        pass

    @abstractmethod
    def register_subclass_hook(self, vis, hook_fn):
        """Abstract method that registers the given hook_fn to some parameters.

        Args:
            vis (bool): Boolean set to true if image only embedding.
            hook_fn (hook function): The hook function to register.
        """
        pass

    @abstractmethod
    def is_input_image(self, input):
        """A function that determines whether the input is an image embedding.

        Args:
            input (tensor): Tensor describing the input
        """
        pass
