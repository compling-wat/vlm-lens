"""minicpm.py.

File for providing the MiniCPM model implementation.
"""
import logging
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import ModelBase
from .config import Config, ModelSelection


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

    def run(self):
        """Runs the model."""
        logging.debug('Building messages ...')
        msgs = []
        for image_path in os.listdir(self.config.input_dir):
            image = Image.open(os.path.join(self.config.input_dir, image_path)).convert('RGB')
            msgs.append({
                'role': 'user',
                'content': [image, self.config.prompt]
            })

        logging.debug('Running MiniCPM model ...')
        self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True),
            )

        logging.debug('Saving states ...')
        self.save_states()
