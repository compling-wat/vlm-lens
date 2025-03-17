"""janus.py.

File for providing the Janus model implementation.
"""
import sys
import os
import torch

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from .base import ModelBase
from .config import Config

class JanusModel(ModelBase):
    """Janus model implementation."""

    def __init__(self, config: Config):
        """Initialization of the Janus model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def _init_processor(self) -> None:
        """Initialize the self.processor by loading from the path."""
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)

    def forward(self, input):
        """Given a list of inputs from the Janus processor, do a forward pass."""

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**input)

        # run the model to get the response
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

    def _call_processor(self):
        """Call the processor with the prompt string and input images to generate the embeddings."""

        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        

    def load_input_data(self):
        """
            Load multiple image inputs and prompts for Janus in a batch, 
            returning the processor outputs as a single batch dictionary.
        """

        # Getting paths for all images under the input_dir
        img_paths = [
            os.path.join(self.config.input_dir, img) 
            for img in os.listdir(self.config.input_dir)
        ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conversations = []

        for img_path in img_paths:
            conversations.extend([  
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{self.config.prompt}",
                    "images": [img_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                },
            ])


        # Load all images as a batch
        imgs = load_pil_images(conversations)  # Ensure this function handles multiple images

        # Process all images and conversations in a single batch call
        batched_output = self.processor(
            images=imgs,  
            conversations=conversations,  
            return_tensors='pt',  
            padding=True,  
            force_batchify=True  # Ensure batch processing
        ).to(device)


        return batched_output  # Single output instead of multiple individual ones
        