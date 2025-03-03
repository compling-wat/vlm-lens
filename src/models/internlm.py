"""internlm.py.

File for providing the InternLM-XComposer model implementation.
"""
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from typing import Dict
from PIL import Image

from .base import ModelBase
from .config import Config

import os
import logging
import torch


class InternLMModel(ModelBase):
    """InternLM model implementation."""

    def __init__(self, config: Config):
        """Initialization of the InternLM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        )


    def _init_processor(self):
        """Overridden function to instantiate the model's processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.tokenizer = self.processor

    def _generate_processor_args(self, prompt):
        return 

    # def forward(self, config: Config):
    #     logging.debug('Starting forward pass')
    #     imgs = [Image.open(os.path.join(config.input_dir, img)).convert('RGB')
    #             for img in os.listdir(config.input_dir)]
        
    #     self.processor()
        
    #     with torch.autocast(device_type='cuda', dtype=torch.float16):
    #         _, _ = self.model.chat(self.processor,
    #                                 [config.prompt for _ in imgs], 
    #                                 imgs, 
    #                                 do_sample=False, 
    #                                 num_beams=3, 
    #                                 use_meta=True)
            
    #     logging.debug('Completed forward pass...')

    # def load_input_data(self, config: Config) -> Dict[str, torch.Tensor]:
    #     """Overridden function to load input data."""
    #     logging.debug('Loading data')

    #     # Set build flags
    #     img_flag = hasattr(config, 'input_dir')
    #     txt_flag = hasattr(config, 'prompt')

    #     # Check if there is no input data
    #     if not img_flag and not txt_flag:
    #         raise RuntimeError('No input data was provided')
        
    #     # Load processor and build prompt
    #     self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)


    #     # Prepare input
    #     logging.debug('Generating embeddings')
    #     if img_flag:
    #         images = [
    #             Image.open(os.path.join(config.input_dir, img)).convert('RGB')
    #             for img in os.listdir(config.input_dir)
    #             ]
    #     # elif txt_flag:

    #     return 