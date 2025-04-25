"""janus.py.

File for providing the Janus model implementation.
"""
import torch
from Janus.janus.models import VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

from src.models.base import ModelBase
from src.models.config import Config


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
        self.is_pro = 'pro' in self.model_path.lower()
        self.is_flow = 'flow' in self.model_path.lower()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model.to(torch.bfloat16)

    def _init_processor(self) -> None:
        """Initialize the Janus processor."""
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
        user_tag = '<|User|>' if self.is_pro else 'User'
        assistant_tag = '<|Assistant|>' if self.is_pro else 'Assistant'

        conversation = [
            {
                'role': user_tag,
                'content': f'<image_placeholder>\n{self.config.prompt}',
                'images': [img_path]
            },
            {
                'role': assistant_tag,
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
            attention_mask=input.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
