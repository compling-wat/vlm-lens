"""janus.py.

File for providing the Janus model implementation.
"""
import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.models.base import ModelBase
from src.models.config import Config

# import Janus as a module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Janus'))
# require this import to force the models script to load
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


class JanusModel(ModelBase):
    """Janus model implementation."""

    def __init__(self, config: Config):
        """Initialize the Janus model.

        Args:
            config (Config): Parsed config.
        """
        super().__init__(config)

    def _load_specific_model(self):
        """Populate self.model with the specified Janus model."""

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # set the attention implementation to eager if it's cpu
        # to whatever we set it to under model if provided
        # or to whatever is the default
        if self.config.device.type == 'cpu':
            config.language_config._attn_implementation = 'eager'
        elif hasattr(self.config, 'model') and 'attn_implementation' in self.config.model:
            config.language_config._attn_implementation = self.config.model['attn_implementation']

        model_args = getattr(self.config, 'model', {})
        if "torch_dtype" in model_args and model_args["torch_dtype"] != "auto":
            model_args["torch_dtype"] = getattr(torch, model_args["torch_dtype"])

        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            **model_args
        )

    def _init_processor(self) -> None:
        """Initialize the Janus processor."""
        self.processor = VLChatProcessor.from_pretrained(self.model_path)

    def _generate_prompt(self):
        """Generates the prompt string with the input messages.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _generate_processor_output(self, prompt, img_path):
        """Override the base function to produce processor arguments for Janus."""

        if img_path is None:
            conversation = [
                {
                    'role': 'User',
                    'content': prompt,
                    'images': []
                },
                {
                    'role': 'Assistant',
                    'content': ''
                }
            ]
        else:
            conversation = [
                {
                    'role': 'User',
                    'content': f'<image_placeholder>\n{prompt}',
                    'images': [img_path]
                },
                {
                    'role': 'Assistant',
                    'content': ''
                }
            ]

        return self.processor(
            conversations=conversation,
            images=load_pil_images(conversation),
            force_batchify=True
        )

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data: The given data tensor.
        """
        data = data.to(self.config.device)

        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**data)
            _ = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=data.attention_mask
            )
            
        logging.debug('Completed forward pass...')
