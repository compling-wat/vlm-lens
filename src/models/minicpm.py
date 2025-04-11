"""minicpm.py.

File for providing the MiniCPM model implementation.
"""
import logging
import os
from typing import Tuple, TypeAlias

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from .base import ModelBase
from .config import Config, ModelSelection

ModelInput: TypeAlias = Tuple[str, str, BatchFeature]


class MiniCPMModel(ModelBase):
    """MiniCPM model implementation."""

    def __init__(self, config: Config):
        """Initialization of the minicpm model.

        This makes sure to set the model name, path and config.

        Args:
            config (Config): Parsed config
        """
        self.model_name = ModelSelection.MINICPM
        self.config = config

        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16
        ) if hasattr(self.config, 'model') else (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16
            )
        )
        self.model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def _init_processor(self) -> None:
        """Initialize the self.processor by loading from the path."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
            )

    def _init_tokenizer(self) -> None:
        """Initialize the self.tokenizer by loading from the path."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
            )

    def _forward(self, data: BatchFeature):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        self._init_tokenizer()

        with torch.no_grad():
            _ = self.model.chat(
                # from huggingface doc, the image is passed in the msgs
                # the image should be left as None
                image=None,
                msgs=data,
                tokenizer=self.tokenizer
                )

        logging.debug('Completed forward pass...')

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

    def run(self):
        """Runs the model."""
        logging.debug('Building messages ...')

        input_data = []
        for image_path in os.listdir(self.config.input_dir):
            image = Image.open(os.path.join(self.config.input_dir, image_path)).convert('RGB')
            input_data.append((
                os.path.join(self.config.input_dir, image_path),
                self.config.prompt,
                [{
                    'role': 'user',
                    'content': [image, self.config.prompt]
                }]
            ))

        for input in input_data:
            self._hook_and_eval(input)

        self._cleanup()
