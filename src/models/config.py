"""config.py.

This module provides a config class to be used for both the parser as well as
for providing the model specific classes a way to access the parsed arguments.
"""
import argparse
import logging
import os
from enum import Enum
from typing import List

import regex as re
import yaml


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""
    LLAVA = 'llava'
    QWEN = 'qwen'
    CLIP = 'clip'


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
            default='./data',
            help='The specified input directory to read data from'
        )
        # TODO: Add in a check to make sure that the output directory exists
        parser.add_argument(
            '-o',
            '--output-dir',
            type=str,
            default='./output_dir',
            help='The specified output directory to save the tensors to'
        )

        args = parser.parse_args()

        # the set of potential keys should be defined by the config + any
        # other special ones here (such as the model args)
        config_keys = list(args.__dict__.keys())
        config_keys.append('model')
        config_keys.append('prompt')
        config_keys.append('text_prompts')
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
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)

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

        self.filters = self._build_filters()
        logging.debug(f'Filters are {self.filters}')

    def _build_filters(self) -> List[dict]:
        """Creates a list of filters to be set and used later during inference.

        Returns:
            List[dict]: The list of filters.
        """
        # set the defaults that we'll use in all cases
        default_prompt = self.prompt if hasattr(self, 'prompt') else None
        default_input_dir = (
            self.input_dir if hasattr(self, 'input_dir') else None
        )
        default_img_filter = '.*'

        # if there is no text_prompts specified, simply return the single
        # default filter
        if not hasattr(self, 'text_prompts'):
            return [
                self._build_single_filter(
                    name='default',
                    prompt=default_prompt,
                    input_dir=default_input_dir,
                    img_filter=default_img_filter
                )
            ]

        # now we will populate a list of possible filters, which will be
        # dictionaries with the fields of: filter_name, prompt and image_paths
        filters = []

        # first, let's preprocess text_prompts to collapse everything
        # into a single dictionary
        collapsed_dict = self._collapse_list_to_dict(self.text_prompts)
        logging.debug(f'Text prompts: {collapsed_dict}')

        for name, filter_dict in collapsed_dict.items():
            filter_dict = self._collapse_list_to_dict(filter_dict)
            filters.append(
                self._build_single_filter(
                    name=name,
                    prompt=(
                        filter_dict['prompt']
                        if 'prompt' in filter_dict.keys() else
                        default_prompt
                    ),
                    input_dir=(
                        filter_dict['input_dir']
                        if 'input_dir' in filter_dict.keys() else
                        default_input_dir
                    ),
                    img_filter=(
                        filter_dict['filter']
                        if 'filter' in filter_dict.keys() else
                        default_img_filter
                    )
                )
            )

        return filters

    def _collapse_list_to_dict(self, list_of_dicts: List[dict]) -> dict:
        """Takes a list of dictionaries and collapses them to a single dict.

        Args:
            list_of_dicts (List[dict]): List of dictionaries to collapse.

        Returns:
            dict: Single collapsed dictionary.
        """
        collapsed_dict = {}
        for dict in list_of_dicts:
            collapsed_dict.update(dict)
        return collapsed_dict

    def _build_single_filter(
        self,
        name: str,
        prompt: str,
        input_dir: str,
        img_filter: str
    ) -> dict:
        """Builds a single filter based on input.

        Args:
            name (str): The name of the filter.
            prompt (str): The prompt string to use.
            input_dir (str): The input images directory.
            img_filter (str): The filters to pattern match to.

        Returns:
            dict: The single filter dictionary.
        """
        if not input_dir and not prompt:
            raise ValueError(
                f'The {name} filter specified has no input directory nor '
                'prompt specified'
            )

        # text-only input case
        if not input_dir:
            return {
                'name': name,
                'prompt': prompt
            }

        # now we know that the input directory is specified
        # so we take a look through all the images in the input directory
        # and add those paths to image_paths
        image_paths = []

        # defined variables to be used within matches_img_filter
        image_exts = [
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'
        ]
        img_filter_regex = re.compile(img_filter)

        def matches_img_filter(file_path: str) -> bool:
            """Returns whether the file path matches the given regex filter.

            Args:
                file_path (str): The image filename.

            Returns:
                bool: Whether the file path matches the given regex
            """
            filename, ext = os.path.splitext(file_path)
            return (
                img_filter_regex.fullmatch(filename) and
                ext.lower() in image_exts
            )

        image_paths = [
            os.path.join(input_dir, img_path)
            for img_path in filter(
                lambda file_path: matches_img_filter(file_path),
                os.listdir(input_dir)
            )
        ]

        logging.debug(f'{name} filter has {image_paths} as its image paths')

        if len(image_paths) <= 0:
            raise ValueError(
                f'Image directory {input_dir} matched no files with '
                f'pattern {img_filter}'
            )

        # finally, build the final dictionary
        filter_out = {
            'name': name,
            'images_path': image_paths
        }

        if prompt:
            filter_out['prompt'] = prompt

        return filter_out

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
