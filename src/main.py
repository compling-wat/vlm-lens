"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""

import logging

from models import llava, qwen
from models.base import ModelBase, ModelSelection
from models.config import Config


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
    logging.debug(get_model(config.architecture, config.model_path, config))
