"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""

import logging
import os
from abc import ABC, abstractmethod

import torch
from PIL import Image
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

    def _generate_state_hook(self, name: str):
        """Generates the state hook depending on the embedding type.

        Args:
            name (str): The module name.

        Returns:
            hook function: The hook function to return.
        """
        def generate_states_hook(module, input, output):
            """Hook handle function that saves the embedding output to a tensor.

            Args:
                module: The module that save its hook on.
                input: The input used.
                output: The embeddings to save.
            """
            # for each module, we'll save its output into
            self.states[name] = output

        return generate_states_hook

    def _register_module_hooks(self):
        """Register the generated hook function to the modules in the config."""
        # set the states to a dictionary such that we can write to it
        # and later on save from all these states
        self.states = {}

        # create a flag to warn the user if there were no hooks registered
        registered_module = False

        for name, module in self.model.named_modules():
            if self.config.matches_module(name):
                registered_module = True
                module.register_forward_hook(self._generate_state_hook(name))
                logging.debug(f'Registered hook to {name}')

        if not registered_module:
            raise RuntimeError(
                'No hooks were registered. Double-check the configured modules.'
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
        if len(self.states.items()) == 0:
            raise RuntimeError('No embedding states were saved')

        for name, state in self.states.items():
            torch.save(
                state,
                os.path.join(
                    self.config.output_dir,
                    f'state_{name}_{self.config.architecture}.pt'
                )
            )

    def load_image_data(self, config: Config) -> torch.Tensor:
        """From a configuration, loads the input image data.

        Args:
            config (Config): The configuration given with image input data
            information.
            model (ModelBase): The model to use for generating the processor.

        Returns:
            torch.Tensor: The data as a torch tensor.
        """
        logging.debug('Loading data...')
        imgs = [
            Image.open(
                os.path.join(config.input_dir, img)
            ).convert('RGB')
            for img in os.listdir(config.input_dir)
        ]

        logging.debug('Generating image prompt embeddings')
        img_data = imgs
        img_msgs = [{
            'role': 'user',
            'content': [
                {
                    'type': 'image'
                },
                {
                    'type': 'text',
                    'text': config.prompt
                },
            ],
        }]

        img_prompt = self.processor.apply_chat_template(
            img_msgs,
            add_generation_prompt=True
        )
        img_inputs = self.processor(
            images=img_data,
            text=[img_prompt for _ in range(len(img_data))],
            return_tensors='pt'
        )

        # TODO: to turn into text + image with "<xxx>" labels

        return img_inputs
