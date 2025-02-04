"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""

import logging
import os
from abc import ABC, abstractmethod

import torch
from transformers import AutoProcessor

from .config import Config


class ModelBase(ABC):
    """Provides an abstract base class for everything to implement."""

    def __init__(self, config: Config):
        """Initialization of the model base class.

        Args:
            config (Config): Parsed config.
        """
        assert self.model_path is not None
        self.config = config
        self.load_model()
        self.register_hook()

    def load_model(self):
        """Loads the model and sets the processor from the loaded model."""
        logging.debug(
            f'Loading model {self.model_name.value}; {self.model_path}'
        )
        self.load_specific_model()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    @abstractmethod
    def load_specific_model(self):
        """Abstract method that loads the specific model."""
        pass

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

    def register_hook(self):
        """Registers the hook depending on the embedding setting.

        Args:
            vis (bool): Set to true in the image only embedding setting.
        """
        logging.debug('Generating hook function')
        self.hook = self.generate_state_hook(self.config.vis)
        self.register_subclass_hook(self.config.vis, self.hook)

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

    def forward(self, data: torch.Tensor):
        """Given some data, performs a single forward pass.

        Args:
            data (torch.Tensor): The input data tensor
        """
        logging.debug('Starting forward pass')
        if self.config.vis:
            with torch.no_grad():
                _ = self.model(**data)
        else:
            assert False

        logging.debug('Completed forward pass...')

    def save_states(self):
        """Saves the states to pt files."""
        torch.save(
            self.vis_image_states,
            os.path.join(
                self.config.output_dir,
                f'visual_tensor_{self.model_name}.pt'
            )
        )
        if not self.config.vis:
            torch.save(
                self.txt_image_states,
                os.path.join(
                    self.config.output_dir,
                    f'text_tensor_{self.model_name}.pt'
                )
            )
