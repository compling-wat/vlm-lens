"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""
import logging

from models import llava, qwen
from models.base import ModelBase
from models.config import Config, ModelSelection


def get_model(
    model_arch: ModelSelection,
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
            return llava.LlavaModel(config)
        case ModelSelection.QWEN:
            return qwen.QwenModel(config)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )

    model = get_model(config.architecture, config)
    model.forward(model.load_input_data())
    model.save_states()
