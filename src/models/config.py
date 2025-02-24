"""config.py.

This module provides a config class to be used for both the parser as well as
for providing the model specific classes a way to access the parsed arguments.
"""
import argparse
from enum import Enum

import regex as re
import yaml


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""
    LLAVA = 'llava'
    QWEN = 'qwen'


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
        # TODO: Add in a check to make sure that the output directory exists
        parser.add_argument(
            '-o',
            '--output-dir',
            type=str,
            help='The specified output directory to save the tensors to'
        )

        args = parser.parse_args()

        # the set of potential keys should be defined by the config + any
        # other special ones here (such as the model args)
        config_keys = list(args.__dict__.keys())
        config_keys.append('model')
        config_keys.append('prompt')
        config_keys.append('modules')

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

        # do an early return if we don't need the modules
        self.log_named_modules = (
            hasattr(self, 'log_named_modules') and self.log_named_modules
        )
        if self.log_named_modules:
            return

        assert hasattr(self, 'modules') and self.modules is not None, (
            'Must declare at least one module.'
        )
        self.modules = [re.compile(module) for module in self.modules]

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
