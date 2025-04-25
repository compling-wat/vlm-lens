"""auto.py.

File for providing model implementations for any models using AutoModel.
"""

import logging

import cv2
import torch
from transformers import (AutoModelForVision2Seq, AutoTokenizer,
                          CLIPImageProcessor)

from src.models.base import ModelBase
from src.models.config import Config


class GlammModel(ModelBase):
    """Glamm model implementation."""

    def __init__(self, config: Config):
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='right',
            use_fast=False
        )

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            AutoModelForVision2Seq.from_pretrained(
                self.model_path
            )
        )

    def _init_processor(self) -> None:
        """Set the self.processor to follow the example given.

        This should follow the processor setting and tokenizers under:
        https://github.com/mbzuai-oryx/groundingLMM/blob/main/app.py
        """
        logging.debug(
            f'GLAMM has vision tower: {self.model.config.vision_tower}'
        )
        self.processor = CLIPImageProcessor.from_pretrained(
            self.model.config.vision_tower
        )

    def _generate_prompt(self) -> str:
        """Generates the GLaMM model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        return self.config.prompt

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor argument to be input into the processor.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor arguments per image and prompt.
        """
        if img_path is None:
            raise ValueError('GLAMM cannot have text-only generation.')

        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        global_enc_image = self.processor.preprocess(
            image_np, return_tensors='pt'
        )['pixel_values'][0].unsqueeze(0).bfloat16()

        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids

        return {
            'global_enc_image': global_enc_image,
            'input_ids': input_ids,
            'bboxes': None  # set this to none for now
        }

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        with torch.no_grad():
            _ = self.model(
                images=data['global_enc_image'].to(self.config.device),
                input_ids=data['input_ids'].to(self.config.device),
                bboxes=data['bboxes']
            )
        logging.debug('Completed forward pass')
