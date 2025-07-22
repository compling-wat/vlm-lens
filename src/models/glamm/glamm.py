"""glamm.py.

File for providing model implementations for any models using AutoModel.
"""

import logging

import cv2
import torch
import torch.nn.functional as F
from transformers import (AutoModelForVision2Seq, AutoTokenizer,
                          CLIPImageProcessor)

from src.models.base import ModelBase
from src.models.config import Config
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'groundingLMM'))

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.generate_utils import center_crop, create_feathered_mask
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tools.markdown_utils import (markdown_default, examples, title, description, article, process_markdown, colors,
                                  draw_bbox, ImageSketcher)

def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    """Preprocess function."""
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x


def prepare_model_for_inference(model, args):
    """Initialize vision tower."""
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args['vision_tower']
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args['local_rank'])
    model = model.bfloat16().cuda()
    return model


class GlammModel(ModelBase):
    """Glamm model implementation."""

    def __init__(self, config: Config):
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        # set up tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            model_max_length=self.config.model['model_max_length'],
            padding_side="right",
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.config.model['bbox_token_idx'] = self.tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
        self.config.model['seg_token_idx'] = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.config.model['bop_token_idx'] = self.tokenizer("<p>", add_special_tokens=False).input_ids[0]
        self.config.model['eop_token_idx'] = self.tokenizer("</p>", add_special_tokens=False).input_ids[0]

        model_args = {
            "seg_token_idx": self.config.model['seg_token_idx'],
            "bbox_token_idx": self.config.model['bbox_token_idx'],
            "eop_token_idx": self.config.model['eop_token_idx'],
            "bop_token_idx": self.config.model['bop_token_idx'],
        }

        self.model = GLaMMForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **model_args
        )
        self.model = prepare_model_for_inference(self.model, self.config.model)

    def _init_processor(self) -> None:
        """Set the self.processor to follow the example given.

        This should follow the processor setting and tokenizers under:
        https://github.com/mbzuai-oryx/groundingLMM/blob/main/app.py
        """

        processor = {
            "global_enc_processor": CLIPImageProcessor.from_pretrained(self.config.model['vision_tower']),
            "grounding_transform": ResizeLongestSide(self.config.model['image_size'])
        }
        self.processor = processor

    def _generate_prompt(self) -> str:
        """Generates the GLaMM model prompt which will not use the chat template.

        Returns:
            str: The prompt to return, set by the config.
        """
        prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n{self.config.prompt}"
        if self.config.model['use_mm_start_end']:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return prompt

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
        orig_h, orig_w = image_np.shape[:2]
        original_size_list = [(orig_h, orig_w)]

        # Global encoder
        global_enc_image = self.processor["global_enc_processor"].preprocess(
            image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().bfloat16()

        # Grounding encoder
        grounding_input = self.processor["grounding_transform"].apply_image(image_np)
        resize_list = [grounding_input.shape[:2]]
        grounding_enc_image = grounding_enc_processor(
            torch.from_numpy(grounding_input).permute(2, 0, 1).contiguous()
        ).unsqueeze(0).cuda().bfloat16()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).cuda()

        return {
            "input_ids": input_ids,
            "global_enc_image": global_enc_image,
            "grounding_enc_image": grounding_enc_image,
            "resize_list": resize_list,
            "original_size_list": original_size_list,
            "bboxes": None
        }

    def _forward(self, data):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        with torch.no_grad():
            output_ids, _ = self.model.evaluate(
                data["global_enc_image"],
                data["grounding_enc_image"],
                data["input_ids"],
                data["resize_list"],
                data["original_size_list"],
                max_tokens_new=self.config.forward['max_new_tokens'],
                bboxes=data["bboxes"]
            )
        logging.debug('Completed forward pass')
