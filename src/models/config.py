"""config.py.

This module provides a config class to be used for both the parser as well as
for providing the model specific classes a way to access the parsed arguments.
"""
import argparse
import logging
import os
from enum import Enum
from typing import List, Optional

import regex as re
import torch
import yaml


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""
    LLAVA = 'llava'
    QWEN = 'qwen'
    CLIP = 'clip'
    GLAMM = 'glamm'
    JANUS = 'janus'
    BLIP2 = 'blip2'
    MOLMO = 'molmo'
    INTERNVL = 'internvl'
    MINICPM = 'minicpm'


class Config():
    """Config class for both yaml and cli arguments."""

    def __init__(self):
        """Verifies the passed arguments while populating config fields."""
        # Initiate parser and parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-c',
            '--config',
            type=str,
            help=''
        )

        model_sel = [model.value for model in list(ModelSelection)]
        parser.add_argument(
            '-a',
            '--architecture',
            type=ModelSelection,
            choices=list(ModelSelection),
            metavar=f'{model_sel}',
            help='The model architecture family to extract the embeddings from'
        )
        parser.add_argument(
            '-m',
            '--model-path',
            type=str,
            help='The specific model path to extract the embeddings from'
        )
        parser.add_argument(
            '-d',
            '--debug',
            default=None,
            action='store_true',
            help='Print out debug statements'
        )
        parser.add_argument(
            '-l',
            '--log_named_modules',
            default=None,
            action='store_true',
            help='Logs the named modules for the specified model'
        )
        # TODO: Add in a check to make sure that the input directory exists
        parser.add_argument(
            '-i',
            '--input-dir',
            type=str,
            help='The specified input directory to read data from'
        )
        parser.add_argument(
            '-o',
            '--output-db',
            type=str,
            help=(
                'The specified output database to save the tensors to, '
                'defaults to embedding.db'
            )
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='Specify the device to send tensors and the model to'
        )

        # only parse the args that we know, and throw out what we don't know
        args = parser.parse_known_args()[0]

        # the set of potential keys should be defined by the config + any
        # other special ones here (such as the model args)
        config_keys = list(args.__dict__.keys())
        config_keys.append('model')
        config_keys.append('prompt')
        config_keys.append('modules')
        config_keys.append('forward')

        # first read the config file and set the current attributes to it
        # then parse through the other arguments as that's what we want use to
        # override the config file if supplied
        if args.config:
            with open(args.config, 'r') as file:
                data = yaml.safe_load(file)

            for key in config_keys:
                if key in data.keys():
                    setattr(self, key, data[key])

        # now we take all the arguments we want and we copy it over!
        for key, value in args._get_kwargs():
            if value is not None:
                setattr(self, key, value)

        # we set the debug flag to False if it doesn't exist
        # And to whatever we would normally set it to otherwise
        self.debug = (
            hasattr(self, 'debug') and self.debug
        )
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        # require that the architecture and the model path to exist
        assert hasattr(self, 'architecture') and hasattr(self, 'model_path'), (
            'Fields `architecture` and `model_path` in yaml config must exist, '
            'otherwise, --architecture and --model-path must be set'
        )

        # change the architecture type to an enum
        if not isinstance(self.architecture, ModelSelection):
            assert self.architecture in model_sel, (
                f'Architecture {self.architecture} not supported, '
                f'use one of {model_sel}'
            )
            self.architecture = ModelSelection(self.architecture)

        if hasattr(self, 'model'):
            model_mapping = {}
            for mapping in self.model:
                model_mapping = {**model_mapping, **mapping}
            self.model = model_mapping

        if hasattr(self, 'forward'):
            forward_mapping = {}
            for mapping in self.forward:
                forward_mapping = {**forward_mapping, **mapping}
            self.forward = forward_mapping

        # do an early return if we don't need the modules
        self.log_named_modules = (
            hasattr(self, 'log_named_modules') and self.log_named_modules
        )
        if self.log_named_modules:
            return

        assert hasattr(self, 'modules') and self.modules is not None, (
            'Must declare at least one module.'
        )
        self.default_modules = self.modules
        self.set_modules(self.modules)

        self.image_paths = []
        self.default_input_dir = (
            self.input_dir
            if hasattr(self, 'input_dir') else
            None
        )
        self.set_image_paths(self.default_input_dir)

        # check if there is no input data
        if not (self.has_images() or hasattr(self, 'prompt')):
            raise ValueError(
                'Input directory was either not provided or empty '
                'and no prompt was provided'
            )

        # now set the default prompt to be used in filters
        self.default_prompt = self.prompt

        # now sets the specific device, first does a check to make sure that if
        # the user wants to use cuda that it is available
        if 'cuda' in self.device and not torch.cuda.is_available():
            raise ValueError('No GPU found for this machine')

        self.device = torch.device(self.device)

        self.DB_TABLE_NAME = 'tensors'
        self.NO_IMG_PROMPT = 'No image prompt'

        # if there is no output database set, use embeddings.db as the default
        if not hasattr(self, 'output_db'):
            self.output_db = 'embeddings.db'

    def has_images(self) -> bool:
        """Returns a boolean for whether or not the input directory has images.

        Returns:
            bool: Whether or not the input directory has images.
        """
        return len(self.image_paths) > 0

    def matches_module(self, module_name: str) -> bool:
        """Returns whether the given module name matches one of the regexes.

        Args:
            module_name (str): The module name to match.

        Returns:
            bool: Whether the given module name matches the config's module
            regexes.
        """
        for module in self.modules:
            if module.fullmatch(module_name):
                return True
        return False

    def set_prompt(self, prompt: str):
        """Sets the prompt for the specific config.

        Args:
            prompt (str): Prompt to set.
        """
        self.prompt = prompt

    def set_modules(self, to_match_modules: List[str]):
        """Sets the modules for the specific config.

        Args:
            to_match_modules (List[str]): The module regexes to match.
        """
        self.modules = [re.compile(module) for module in to_match_modules]

    def set_image_paths(self, input_dir: Optional[str]):
        """Sets the images based on the input directory.

        Args:
            input_dir (Optional[str]): The input directory.
        """
        if input_dir is None:
            return
        # now we take a look through all the images in the input directory
        # and add those paths to image_paths
        image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        self.image_paths = [
            os.path.join(input_dir, img_path)
            for img_path in filter(
                lambda file_path:
                    os.path.splitext(file_path)[1].lower() in image_exts,
                os.listdir(self.input_dir)
            )
        ]
