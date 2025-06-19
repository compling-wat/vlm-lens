"""plm.py.

File for providing the Plm model implementation.
"""
import os
import sys
from dataclasses import fields

from huggingface_hub import login
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from src.models.base import ModelBase
from src.models.config import Config

# import Plm as a module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Plm'))


class PlmModel(ModelBase):
    """Plm model implementation."""

    def __init__(self, config: Config):
        """Initialize the plm model.

        Args:
            config (Config): Parsed config.
        """
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate PLM model.

        Huggingface token is required to get access to the model.
        Replace <HUGGINGFACE_TOKEN> in configs/plm-1b.yaml file with you own hugging face security token.
        Note: 'token' is a general Hugging Face Hub access token, not specific to PLM.
        It enables loading private models or authenticated access.
        See: https://huggingface.co/docs/hub/en/security-tokens
        """
        from apps.plm.generate import (PackedCausalTransformerGenerator,
                                       PackedCausalTransformerGeneratorArgs,
                                       load_consolidated_model_and_tokenizer)
        from core.args import dataclass_from_dict
        from core.transforms.image_transform import get_image_transform

        login(token=self.config.model['token'], new_session=True)
        self.model, self.tokenizer, self.plm_config = load_consolidated_model_and_tokenizer(self.model_path)

        self.transform = get_image_transform(
            vision_input_type=self.plm_config.data.vision_input_type,
            image_res=self.model.vision_model.image_size,
            max_num_tiles=self.plm_config.data.max_num_tiles,
        )

        gen_keys = [f.name for f in fields(PackedCausalTransformerGeneratorArgs)]
        gen_args = {}

        for key in gen_keys:
            if key in self.config.model:
                gen_args[key] = self.config.model[key]

        gen_args['device'] = str(self.config.device)

        self.gen_cfg = dataclass_from_dict(
            PackedCausalTransformerGeneratorArgs,
            gen_args,
            strict=False,
        )

        self.generator = PackedCausalTransformerGenerator(self.gen_cfg, self.model, self.tokenizer)

    def _init_processor(self) -> None:
        """Initialize the plm processor. No processor for plm."""
        return None

    def _generate_prompt(self) -> str:
        """Generates the plm model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _forward(self, data: BatchFeature):
        """Given some input data, performs a single forward pass.

        PLM is not supporting text only generation. If data doesn't contain image or image is None, return None

        Args:
            data: The given data tensor.
        """
        if 'image' not in data or data['image'] is None:
            return None

        image, _ = self.transform(data['image'])
        prompt = [(data['text'], image)]
        return self.generator.generate(prompt)

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor outputs from the prompt and image path.

        PLM currently is not supporting text only generation, be sure to contain both text and image.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        if img_path is None:
            image = None
        else:
            image = Image.open(img_path).convert('RGB')

        return {
            'text': prompt,
            'image': image,
        }
