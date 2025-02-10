"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""

import logging
import os
import warnings
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
        self.model_path = config.model_path
        self.config = config

        # load the specific model
        logging.debug(
            f'Loading model {self.config.architecture.value}; {self.model_path}'
        )
        self._load_specific_model()

        # now set up the modules to register the hook to
        self._register_module_hooks()

        # set the processor based on the model
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # generate and register the forward hook
        logging.debug('Generating hook function')

    @abstractmethod
    def _load_specific_model(self):
        """Abstract method that loads the specific model."""
        pass

    def _generate_state_hook(self):
        """Generates the state hook depending on the embedding type.

        Returns:
            hook function: The hook function to return.
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

        return generate_vis_state_hook

    def _register_module_hooks(self):
        """Register the generated hook function to the modules in the config."""
        self.hook = self._generate_state_hook()

        # create a flag to warn the user if there were no hooks registered
        registered_module = False

        for name, module in self.model.named_modules():
            if self.config.matches_module(name):
                registered_module = True
                module.register_forward_hook(self.hook)
                logging.debug(f'Registered hook to {name}')

        if not registered_module:
            warnings.warn(
                "No hooks were registered. Double-check the model's modules",
                UserWarning
            )

    def forward(self, data: torch.Tensor):
        """Given some data, performs a single forward pass.

        Args:
            data (torch.Tensor): The input data tensor
        """
        logging.debug('Starting forward pass')
        with torch.no_grad():
            _ = self.model(**data)
        logging.debug('Completed forward pass...')

    def save_states(self):
        """Saves the states to pt files."""
        assert hasattr(self, 'vis_image_states'), (
            'Error in registering hook, vis_image_states not found'
        )
        torch.save(
            self.vis_image_states,
            os.path.join(
                self.config.output_dir,
                f'visual_tensor_{self.config.architecture}.pt'
            )
        )
