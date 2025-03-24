"""janus.py.

File for providing the Janus model implementation.
"""
import os

import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

from .base import ModelBase
from .config import Config


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
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def _init_processor(self) -> None:
        """Initialize the Janus processor."""
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)

    def forward(self, input):
        """Perform a forward pass with the given input."""
        inputs_embeds = self.model.prepare_inputs_embeds(**input)
        self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=input.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

    def load_input_data(self):
        """Load and preprocess batch input data from images and prompts."""
        is_pro = 'pro' in self.model_path.lower()

        img_paths = [
            os.path.join(self.config.input_dir, img)
            for img in os.listdir(self.config.input_dir)
        ]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        user_tag = '<|User|>' if is_pro else 'User'
        assistant_tag = '<|Assistant|>' if is_pro else 'Assistant'

        conversations = []
        for img_path in img_paths:
            conversations.extend([
                {
                    'role': user_tag,
                    'content': f'<image_placeholder>\n{self.config.prompt}',
                    'images': [img_path]
                },
                {
                    'role': assistant_tag,
                    'content': ''
                },
            ])

        imgs = load_pil_images(conversations)

        batched_output = self.processor(
            images=imgs,
            conversations=conversations,
            return_tensors='pt',
            padding=True,
            force_batchify=True
        ).to(device)

        return batched_output

    def run(self):
        """Run the model and save the output states."""
        self.forward(self.load_input_data())
        self.save_states()
