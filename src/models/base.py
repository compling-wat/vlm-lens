"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

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

        # set the processor based on the model
        self._load_processor()

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

    def _load_processor(self):
        """Given the model path set, load the processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def forward(self, data: BatchFeature):
        """Given some data, performs a single forward pass.

        Args:
            data (BatchFeature): The input data dictionary
        """
        logging.debug('Starting forward pass')
        self.model.eval()
        with torch.no_grad():
            _ = self.model(**data)
        logging.debug('Completed forward pass...')

    def save_states(self, filter_name: str):
        """Saves the states to pt files.

        Args:
            filter_name (str): The filter name used.
        """
        if len(self.states.items()) == 0:
            raise RuntimeError('No embedding states were saved')

        for name, state in self.states.items():
            filename = (
                f'filter_{filter_name}_{name}_'
                f'{self.config.architecture}.pt'
            )
            torch.save(
                state,
                os.path.join(
                    self.config.output_dir,
                    filename
                )
            )
            logging.debug(f'Finished writing to {filename}')

    def _generate_processor_args(self, img_filter: dict) -> dict:
        """Generate the processor arguments to be input into the processor.

        Args:
            img_filter (dict): The image filter that specifies the image and
                text inputs.

        Returns:
            dict: The processor arguments.
        """
        prompt = self._generate_prompt(img_filter)
        has_images = 'images_path' in img_filter.keys()

        processor_args = {
            'return_tensors': 'pt'
        }

        # only add text if prompt exists
        if prompt:
            processor_args['text'] = (
                [prompt for _ in img_filter['images_path']]
                if has_images else prompt
            )

        # only add images if it exists
        if has_images:
            processor_args['images'] = [
                Image.open(img_path).convert('RGB')
                for img_path in img_filter['images_path']
            ]
        return processor_args

    def _generate_prompt(
        self,
        img_filter: dict,
        add_generation_prompt: bool = True
    ) -> str:
        """Generates the prompt string with the input messages.

        Args:
            img_filter (dict): The image filter that specifies the image and
                text inputs.
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
        if len(img_filter['images_path']) > 0:
            input_msgs_formatted[0]['content'].append({
                'type': 'image'
            })

        # add the prompt if it exists
        if 'prompt' in img_filter.keys():
            input_msgs_formatted[0]['content'].append({
                'type': 'text',
                'text': img_filter['prompt']
            })

        # apply the chat template to get the prompt
        return self.processor.apply_chat_template(
            input_msgs_formatted,
            add_generation_prompt=add_generation_prompt
        )

    def _call_processor(self, img_filter: dict) -> BatchFeature:
        """Call the processor with prompt and input images to generate embeddings.

        Args:
            img_filter (dict): The image filter that specifies the image and
                text inputs.

        Returns:
            BatchFeature: The batch feature object with the input data.
        """
        logging.debug('Generating embeddings...')

        # generate the inputs
        return self.processor(**self._generate_processor_args(
            img_filter=img_filter
        ))

    def load_input_data(self) -> List[Tuple[str, BatchFeature]]:
        """From a configuration, loads the input image and text data.

        Returns:
            List[Tuple[str, BatchFeature]]: The list of input data as either
                a torch.Tensor or a Dict alongside with its filter name.
        """
        return [
            (img_filter['name'], self._call_processor(img_filter))
            for img_filter in self.config.filters
        ]

    def run(self) -> None:
        """Get the hidden states from the model and saving them."""
        batch_features = self.load_input_data()
        for filter_name, batch_feature in batch_features:
            self.forward(batch_feature)
            self.save_states(filter_name)
