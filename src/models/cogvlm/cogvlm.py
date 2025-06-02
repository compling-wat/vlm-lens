"""cogvlm.py.

File for providing the CogVLM model implementation.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
# from transformers.feature_extraction_utils import BatchFeature

from src.models.base import ModelBase
from src.models.config import Config

class CogVLMModel(ModelBase):
    """CogVLM model implementation."""

    def __init__(self, config: Config):
        """Initialization of the CogVLM model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)
    
    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModelForCausalLM.from_pretrained(
                self.model_path
            )
        )
    
    def _init_processor(self) -> None:
        """Initialize the CogVLM processor.

        Follows the processor setting and tokenizers under:
        https://huggingface.co/THUDM/cogvlm-chat-hf
        """
        self.processor = None  # no intended processor here
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', legacy=True)
    
    def _generate_prompt(self) -> str:
        """Generates the CogVLM model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt
    
    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        if img_path is None:
            raise ValueError('CogVLM cannot have text-only generation.')
        
        image = Image.open(img_path).convert('RGB')

        # chat mode
        input_ids = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])
        return {
            'input_ids': input_ids['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': input_ids['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': input_ids['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[input_ids['images'][0].to('cuda').to(torch.bfloat16)]],
            'max_length': 2048,
            'do_sample': False
        }
