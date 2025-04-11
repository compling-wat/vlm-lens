"""molmo.py.

File for providing the Molmo model implementation.
"""
import logging
import os
from typing import List, Tuple, TypeAlias

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.feature_extraction_utils import BatchFeature

from .base import ModelBase
from .config import Config, ModelSelection

ModelInput: TypeAlias = Tuple[str, str, BatchFeature]


class MolmoModel(ModelBase):
    """Molmo model implementation."""

    def __init__(self, config: Config):
        """Initialization of the molmo model.

        This makes sure to set the model name, path and config.

        Args:
            config (Config): Parsed config
        """
        self.model_name = ModelSelection.MOLMO
        self.config = config

        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **self.config.model, trust_remote_code=True
        ) if hasattr(self.config, 'model') else (
            AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        )

    def _init_processor(self) -> None:
        """Initializes the processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
        )

    def _call_processor(self) -> BatchFeature:
        """Call processor to process the input data."""
        logging.debug('Generating embeddings ... ')

        # generate the inputs
        inputs = self.processor.process(
            text=self.config.prompt,
            images=[Image.open(os.path.join(self.config.input_dir, image)) for image in os.listdir(self.config.input_dir)],
            )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        return inputs

    def _load_input_data(self) -> List[ModelInput]:
        """From a configuration, loads the input image and text data.

        For each prompt and input image, create a separate batch feature that
        will be ran separately and saved separately within the database.

        Returns:
            List[ModelInput]: List of input data, this input data is made of
                a tuple of strings (first an image path, then a prompt) and
                a batch feature which is either a torch.Tensor or a dictionary.
        """
        # by default use the processor, which may not exist for each model
        logging.debug('Generating embeddings through its processor...')

        input_data = []

        for image_path in self.config.image_paths:
            # generate the inputs
            inputs = self.processor.process(
                text=self.config.prompt,
                images=[Image.open(image_path).convert('RGB')],
            )

            inputs_on_device = {k: v.to(self.model.device).unsqueeze(0)
                                for k, v in inputs.items()
                                }

            # add the input data to the list
            input_data.append((image_path,
                               self.config.prompt,
                               inputs_on_device
                               ))

        return input_data

    def _hook_and_eval(self, input: ModelInput):
        """Given some input, performs a single forward pass.

        Args:
            input (ModelInput): The tuple of the image path, prompt and
                input data dictionary.
        """
        logging.debug('Starting forward pass')
        self.model.eval()

        image_path, prompt, data = input

        # now set up the modules to register the hook to
        hooks = self._register_module_hooks(image_path, prompt)

        # then ensure that the data is correct
        self._forward(data)

        for hook in hooks:
            hook.remove()
        logging.debug('Unregistered all hooks..')

    def _forward(self, data: BatchFeature):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        with torch.no_grad():
            _ = self.model.generate_from_batch(
                data,
                GenerationConfig(max_new_tokens=200, stop_strings='<|endoftext|>'),
                tokenizer=self.processor.tokenizer)
        logging.debug('Completed forward pass...')
