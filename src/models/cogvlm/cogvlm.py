"""cogvlm.py.

File for providing the CogVLM model implementation.
"""
import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

from src.models.base import ModelBase
from src.models.config import Config


class CogVLMModel(ModelBase):
    """CogVLM model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the CogVLM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=self.config.model['low_cpu_mem_usage'],
            trust_remote_code=self.config.model['trust_remote_code']
        ) if hasattr(self.config, 'model') else (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        )

    def _init_processor(self) -> None:
        """Initialize the CogVLM processor.

        Follows the processor setting and tokenizers under:
        https://huggingface.co/THUDM/cogvlm-chat-hf
        """
        self.processor = None  # no intended processor here
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config.model['tokenizer_path'],
            legacy=self.config.model['legacy']
        )

    def _generate_prompt(self, prompt: str) -> str:
        """Generates the CogVLM model prompt which will not use the chat template.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            str: The prompt to return, set by the config.
        """
        return prompt

    def _generate_processor_output(self, prompt: str, img_path: str | None) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.

        Raises:
            ValueError: If the image path is not defined.
        """
        if img_path is None:
            raise ValueError('Define input image directory in model config.')

        image = Image.open(img_path).convert('RGB')

        # build input data
        input_ids = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=prompt,
            history=[],
            images=[image],
            template_version=self.config.model['template_version']
        )

        return {
            'input_ids': input_ids['input_ids'].unsqueeze(0),
            'token_type_ids': input_ids['token_type_ids'].unsqueeze(0),
            'attention_mask': input_ids['attention_mask'].unsqueeze(0),
            'images': input_ids['images'][0].to(torch.bfloat16),
        }

    def _forward(self, data: dict) -> None:
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (dict): The given data tensor.
        """
        gen_kwargs = self.config.forward

        with torch.no_grad():
            _ = self.model.generate(
                input_ids=data['input_ids'].to(self.config.device),
                token_type_ids=data['token_type_ids'].to(self.config.device),
                attention_mask=data['attention_mask'].to(self.config.device),
                images=[[data['images'].to(self.config.device)]],
                **gen_kwargs
            )

        logging.debug('Completed forward pass...')
