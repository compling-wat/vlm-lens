"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""

import logging
import os

import torch
from PIL import Image

from models import llava, qwen
from models.base import ModelBase
from models.config import Config, ModelSelection


def get_model(
    model_arch: ModelSelection,
    model_path: str,
    config: Config
) -> ModelBase:
    """Returns the model based on the selection enum chosen.

    Args:
        model_arch (ModelSelection): ModelSelection enum chosen for the specific
        architecture.
        model_path (str): The specific model within the family architecture of
        model_arch.

    Returns:
        base.ModelBase: A model of type ModelBase which implements the runtime
    """
    match model_arch:
        case ModelSelection.LLAVA:
            return llava.LlavaModel(model_path, config)
        case ModelSelection.QWEN:
            return qwen.QwenModel(model_path, config)


def load_image_data(config: Config, model: ModelBase) -> torch.Tensor:
    """From a configuration, loads the input image data.

    Args:
        config (Config): The configuration given with image input data
        information.
        model (ModelBase): The model to use for generating the processor

    Returns:
        torch.Tensor: The data as a torch tensor
    """
    logging.debug('Loading data...')
    imgs = [
        Image.open(
            os.path.join(config.input_dir, img)
        ).convert('RGB')
        for img in os.listdir(config.input_dir)
    ]

    logging.debug('Generating image prompt embeddings')
    # TODO: remove the self.config.vis -- handle this elsewhere...
    img_data = imgs
    img_msgs = [{
        'role': 'user',
        'content': [
            {
                'type': 'image'
            },
            {
                'type': 'text',
                'text': 'Describe the color in this image in one word.'
            },
        ],
    }]

    img_prompt = model.processor.apply_chat_template(
        img_msgs,
        add_generation_prompt=True
    )
    img_inputs = model.processor(
        images=img_data,
        text=[img_prompt for _ in range(len(img_data))],
        return_tensors='pt'
    )

    return img_inputs


if __name__ == '__main__':
    config = Config()
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )
    model = get_model(config.architecture, config.model_path, config)
    model.forward(load_image_data(config, model))
    model.save_states()

    # TODO: Look at setting the model to eval
    # make sure that the train part doesn't introduce stochasticity
