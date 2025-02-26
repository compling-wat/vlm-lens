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
from transformers.feature_extraction_utils import BatchFeature

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

        # load the processor
        self._init_processor()

        # now set up the modules to register the hook to
        self._register_module_hooks()

        # generate and register the forward hook
        logging.debug('Generating hook function')

    @abstractmethod
    def _load_specific_model(self):
        """Abstract method that loads the specific model."""
        pass

    def _init_processor(self) -> None:
        """Initialize the self.processor by loading from the path."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)

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

    def forward(self, data: BatchFeature):
        """Given some data, performs a single forward pass.

        Args:
            data (BatchFeature): The input data dictionary
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

    def _generate_prompt(self,
                         add_generation_prompt: bool = True
                         ) -> str:
        """Generates the prompt string from the input messages.

        Args:
            add_generation_prompt (bool): Whether to add the generation prompt to the prompt.

        Returns:
            str: The generated prompt with the input text and the image labels.
        """
        logging.debug('Loading data...')

        # build the input dict for the chat template
        input_msgs_formatted = [{
            'role': 'user',
            'content': []
        }]
        # add the image if it exists
        if hasattr(self.config, 'input_dir'):
            input_msgs_formatted[0]['content'].append({
                'type': 'image'
            })
        # add the prompt if it exists
        if hasattr(self.config, 'prompt'):
            input_msgs_formatted[0]['content'].append({
                'type': 'text',
                'text': self.config.prompt
            })

        # apply the chat template to get the prompt
        return self.processor.apply_chat_template(
            input_msgs_formatted,
            add_generation_prompt=add_generation_prompt
        )

    def _call_processor(self) -> BatchFeature:
        """Call the processor with the prompt string and input images to generate the embeddings.

        Returns:
            BatchFeature: The batch feature object with the input data.
        """
        logging.debug('Generating embeddings...')

        prompt_formatted = self._generate_prompt()

        # image-only or image and text
        if hasattr(self.config, 'input_dir'):
            inputs = self.processor(
                images=[
                    Image.open(os.path.join(self.config.input_dir, img)).convert('RGB') for img in os.listdir(self.config.input_dir)
                ],
                text=[prompt_formatted for _ in os.listdir(self.config.input_dir)],
                return_tensors='pt'
            )
        # text-only
        else:
            inputs = self.processor(
                text=prompt_formatted,
                return_tensors='pt'
            )
        return inputs

    def load_input_data(self) -> BatchFeature:
        """From a configuration, loads the input image and text data.

        Returns:
            BatchFeature: The input data as either a torch.Tensor or a Dict.
        """
        # check if there is no input data
        if not hasattr(self.config, 'input_dir') and not hasattr(self.config, 'prompt'):
            raise RuntimeError('No input data was provided')

        # build the input batch features
        inputs = self._call_processor()

        return inputs
