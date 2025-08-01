"""janus.py.

File for providing the Janus model implementation.
"""
import torch
from transformers import JanusForConditionalGeneration, JanusProcessor

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
        # require this import to force the models script to load
        self.model = (
            JanusForConditionalGeneration.from_pretrained(
                self.model_path,
                **self.config.model
            ) if hasattr(self.config, 'model') else
            JanusForConditionalGeneration.from_pretrained(
                self.model_path,
            )
        )
        self.model.to(torch.bfloat16)

    def _init_processor(self) -> None:
        """Initialize the Janus processor."""
        self.processor = JanusProcessor.from_pretrained(self.model_path)

    def _generate_prompt(self, add_generation_prompt=True):
        """Generates the prompt string with the input messages."""
        return self.config.prompt  # We have to delay the process to _generate_processor_output, as we need image path here

    def _generate_processor_output(self, prompt, img_path):
        """Override the base function to produce processor arguments for Janus."""
        # Do the _generate_prompt first
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img_path},
                    {'type': 'text', 'text': prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            generation_mode='text',
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.config.device, dtype=torch.bfloat16)

        return inputs

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data: The given data tensor.
        """
        _ = self.model.generate(**data, max_new_tokens=1, generation_mode='text', do_sample=False)
