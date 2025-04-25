"""janus.py.

File for providing the Janus model implementation.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.models.base import ModelBase
from src.models.config import Config

# import Janus as a module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Janus'))


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
        # require this import to force the models script to load
        from janus.models import MultiModalityCausalLM

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # set the attention implementation to eager if it's cpu
        # to whatever we set it to under model if provided
        # or to whatever is the default
        if self.config.device == torch.device('cpu'):
            config.language_config._attn_implementation = 'eager'
        elif (
            hasattr(self.config, 'model') and
            'attn_implementation' in self.config.model.keys()
        ):
            config.language_config._attn_implementation = \
                    self.config.model['attn_implementation']

        self.model: MultiModalityCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                **self.config.model
            ) if hasattr(self.config, 'model') else
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config
            )
        )
        self.model.to(torch.bfloat16)

    def _init_processor(self) -> None:
        """Initialize the Janus processor."""
        from janus.models import VLChatProcessor

        self.processor = VLChatProcessor.from_pretrained(self.model_path)

    def _generate_prompt(self, add_generation_prompt=True):
        """Generates the prompt string with the input messages.

        Args:
            add_generation_prompt (bool): Whether to add a start token of a bot
                response.
            TODO: move `add_generation_prompt` to the config.

        Returns:
            str: The generated prompt with the input text and the image labels.
        """
        return self.config.prompt

    def _generate_processor_output(self, prompt, img_path):
        """Override the base function to produce processor arguments for Janus."""
        from janus.utils.io import load_pil_images

        conversation = [
            {
                'role': 'User',
                'content': f'<image_placeholder>\n{self.config.prompt}',
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
        inputs_embeds = self.model.prepare_inputs_embeds(**data)
        return self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=data.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
