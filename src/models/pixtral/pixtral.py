"""pixtral.py.

File for providing the Pixtral model implementation.
"""
import logging

from huggingface_hub import snapshot_download
import torch
from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import ImageChunk, TextChunk, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image

from src.models.base import ModelBase
from src.models.config import Config

class PixtralModel(ModelBase):
    """Pixtral model implementation."""

    def __init__(self, config: Config):
        """Initialization of the Pixtral model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""

        snapshot_download(
            repo_id=self.model_path,
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=self.config.download_path,
        )

        self.model = Transformer.from_folder(self.config.download_path, **getattr(self.config, 'model', {}))

    def _generate_prompt(self) -> str:
        """Generates the Pixtral model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _init_processor(self) -> None:
        """Initialize the Pixtral Tokenizer"""
        self.processor = None # no intended processor here
        self.tokenizer = MistralTokenizer.from_file(f"{self.config.download_path}/tekken.json")

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Pixtral uses a specific chat template format with special image tokens.
        
        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str or None): The specified image path, or None for text-only.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """

        user_content = [TextChunk(text=prompt)]
        if img_path is not None:
            image = Image.open(img_path)
            user_content = [ImageChunk(image=image)] + user_content

        completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_content)])
        encoded = self.tokenizer.encode_chat_completion(completion_request)

        res = {
            "input_ids": torch.tensor(encoded.tokens, dtype=torch.long, device=self.model.device),
            "seqlens": [len(encoded.tokens)],
        }

        if img_path is not None:
            res["images"] = [
                torch.tensor(img, device=self.model.device, dtype=self.model.dtype) 
                for img in encoded.images
            ]

        return res

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.
        """
        with torch.no_grad():
            _ = self.model.forward(**data)
        logging.debug('Completed forward pass...')
