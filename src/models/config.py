"""config.py.

This module provides a config class to be used for both the parser as well as
for providing the model specific classes a way to access the parsed arguments.
"""
import argparse
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import regex as re
import torch
import yaml
from datasets import load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # models -> src -> root
sys.path.append(str(PROJECT_ROOT))


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""
    LLAVA = 'llava'
    QWEN = 'qwen'
    CLIP = 'clip'
    GLAMM = 'glamm'
    JANUS = 'janus'
    BLIP2 = 'blip2'
    MOLMO = 'molmo'
    PALIGEMMA = 'paligemma'
    INTERNLM_XC = 'internlm-xcomposer'
    INTERNVL = 'internvl'
    MINICPM = 'minicpm'
    COGVLM = 'cogvlm'
    PIXTRAL = 'pixtral'
    AYA_VISION = 'aya-vision'
    PLM = 'plm'


class Config:
    """Config class for both yaml and cli arguments."""

    def __init__(self,
                 architecture: Optional[str] = None,
                 model_path: Optional[str] = None,
                 module: Optional[str] = None,
                 prompt: Optional[str] = None) -> None:
        """Verifies the passed arguments while populating config fields.

        Args:
            architecture (Optional[str]): The model architecture to use.
            model_path (Optional[str]): The specific model path to use.
            module (Optional[str]): The specific module to extract embeddings from.
            prompt (Optional[str]): The prompt to use for models that require it.

        Raises:
            ValueError: If any required argument is missing.
        """
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
            default=architecture,
            help='The model architecture family to extract the embeddings from'
        )
        parser.add_argument(
            '-m',
            '--model-path',
            type=str,
            default=model_path,
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
            '--log-named-modules',
            default=None,
            action='store_true',
            help='Logs the named modules for the specified model'
        )
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
            default='cuda',
            choices=['cuda', 'cpu'],
            help='Specify the device to send tensors and the model to'
        )
        parser.add_argument(
            '--download-path',
            type=str,
            help='The path where downloaded models should be stored'
        )
        parser.add_argument(
            '--pooling-method',
            type=str,
            default=None,
            choices=['mean', 'max'],
            help='The type of pooling to use for the output embeddings'
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
        config_keys.append('dataset')

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
        assert all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in ('architecture', 'model_path')
        ), (
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

        # if the model is set, make sure that it is a mapping
        if hasattr(self, 'model'):
            model_mapping = {}
            for mapping in self.model:
                model_mapping = {**model_mapping, **mapping}
            self.model = model_mapping

        # if forward is set, make sure that it is a mapping
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

        # override the modules if we have a module passed in
        if module is not None:
            self.modules = [module]
        assert hasattr(self, 'modules') and self.modules is not None, (
            'Must declare at least one module.'
        )
        self.set_modules(self.modules)

        # make sure only one of dataset or input_dir is set
        if hasattr(self, 'dataset') and hasattr(self, 'input_dir'):
            raise ValueError(
                'Only one of `dataset` or `input_dir` can be set, '
                'not both. Please choose one.'
            )

        self.image_paths = []
        if hasattr(self, 'dataset'):
            # Make sure it is a mapping
            ds_mapping = {}
            for mapping in self.dataset:
                ds_mapping = {**ds_mapping, **mapping}

            dataset_path = ds_mapping.get('dataset_path', None)
            local_dataset_path = ds_mapping.get('local_dataset_path', None)

            # Check that the user uses either a local or hosted dataset (not both)
            assert ((dataset_path and not local_dataset_path) or
                    (not dataset_path and local_dataset_path)), (
                'One of `dataset_path` (for hosted datasets) or `local_dataset_path` (for local datasets)'
                'must be set.'
            )

            dataset = None
            dataset_split = ds_mapping.get('dataset_split', None)
            if dataset_path:
                # Dataset is hosted
                logging.debug(f'Loading dataset from {dataset_path} with split={dataset_split}...')
                dataset = load_dataset(dataset_path)

            elif local_dataset_path:
                # Dataset is local
                logging.debug(f'Loading dataset from {local_dataset_path} with split={dataset_split}...')
                dataset = load_from_disk(local_dataset_path)

            dataset = dataset[dataset_split] if dataset_split else dataset

            # Load image dataset
            img_dir = ds_mapping.get('image_dataset_path', None)
            if img_dir:
                logging.debug(
                    f'Locating image dataset from {img_dir}...')

                # Accounts for mapping relative paths as well as filenames
                dataset = dataset.map(
                    lambda row: {'image': os.path.join(img_dir, row['image'])})

                self.image_paths = dataset['image']  # for debug purposes

            self.dataset = dataset

        else:
            self.dataset = None
            self.set_image_paths(self.input_dir
                                 if hasattr(self, 'input_dir') else
                                 None)
        # override the modules if we have a module passed in
        if prompt is not None:
            self.prompt = prompt
        # check if there is no input data
        if not (self.dataset or self.has_images() or hasattr(self, 'prompt')):
            raise ValueError(
                'Input directory was either not provided or empty '
                'and no prompt was provided'
            )

        # now sets the specific device, first does a check to make sure that if
        # the user wants to use cuda that it is available
        if 'cuda' in self.device and not torch.cuda.is_available():
            raise ValueError('Device set to cuda but no GPU found for this machine')

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

    def set_prompt(self, prompt: str) -> None:
        """Sets the prompt for the specific config.

        Args:
            prompt (str): Prompt to set.
        """
        self.prompt = prompt

    def set_modules(self, to_match_modules: List[str]) -> None:
        """Sets the modules for the specific config.

        Args:
            to_match_modules (List[str]): The module regexes to match.
        """
        self.modules = [re.compile(module) for module in to_match_modules]

    def set_image_paths(self, input_dir: Optional[str]) -> None:
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
            os.path.join(root, file_path)
            for root, _, files in os.walk(input_dir)
            for file_path in files
            if os.path.splitext(file_path)[1].lower() in image_exts
        ]
