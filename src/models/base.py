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

        # log the modules -- note that this causes an exit
        if self.config.log_named_modules:
            self._log_named_modules()
            exit(0)

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

    def _log_named_modules(self):
        """Logs the named modules based on the loaded model."""
        file_path = 'logs/' + self.model_path + '.txt'
        directory_path = os.path.dirname(file_path)

        # if the path exists to the file, don't load the model again
        if os.path.isfile(file_path):
            logging.debug(f'Named modules are cached in {file_path}')
            return

        # in which case, we first load the model, then output its modules
        self._load_specific_model()

        # otherwise, we log the output to that file, and creating directories
        # as needed
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(file_path, 'w') as output_file:
            output_file.writelines(
                [f'{name}\n' for name, _ in self.model.named_modules()]
            )

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
        self.model.eval()

        # then ensure that the data is correct
        data.to(self.config.device)

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
                    f"state-{name.replace('.', '_')}-{self.config.architecture.value}.pt"
                )
            )

    def _generate_processor_args(self, prompt) -> dict:
        """Generate the processor arguments to be input into the processor.

        Args:
            prompt (str): The generated prompt string with the input text and the image labels.

        Returns:
            dict: The processor arguments.
        """
        has_images = self.config.has_images()
        processor_args = {
            'text': (
                [prompt for _ in self.config.image_paths]
                if has_images else
                prompt
            ),
            'return_tensors': 'pt'
        }
        if has_images:
            processor_args['images'] = [
                Image.open(img_path).convert('RGB')
                for img_path in self.config.image_paths
            ]
        return processor_args

    def _generate_prompt(self, add_generation_prompt: bool = True) -> str:
        """Generates the prompt string with the input messages.

        Args:
            add_generation_prompt (bool): Whether to add a start token of a bot
                response.
            TODO: move `add_generation_prompt` to the config.

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
        if self.config.has_images():
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
        """Call the processor with prompt and input images to generate embeddings.

        Returns:
            BatchFeature: The batch feature object with the input data.
        """
        logging.debug('Generating embeddings...')

        # format the prompt
        prompt_formatted = self._generate_prompt()

        # generate the inputs
        inputs = self.processor(**self._generate_processor_args(
            prompt=prompt_formatted
        ))

        return inputs

    def load_input_data(self) -> BatchFeature:
        """From a configuration, loads the input image and text data.

        Returns:
            BatchFeature: The input data as either a torch.Tensor or a Dict.
        """
        # build the input batch features
        inputs = self._call_processor()

        return inputs

    def run(self) -> None:
        """Get the hidden states from the model and saving them."""
        # first convert to gpu state
        self.model.to(self.config.device)

        # then run everything else
        self.forward(self.load_input_data())
        self.save_states()
