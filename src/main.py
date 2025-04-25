"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""
import logging
import os
import sys

from models.base import ModelBase
from models.config import Config, ModelSelection

# add on the src directory to the PythonPath
EXC_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(EXC_DIR)
sys.path.append(os.path.join(EXC_DIR, 'src/'))


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
    from models import clip, glamm, janus, llava, qwen

    if model_arch == ModelSelection.LLAVA:
        return llava.LlavaModel(config)
    elif model_arch == ModelSelection.QWEN:
        return qwen.QwenModel(config)
    elif model_arch == ModelSelection.CLIP:
        return clip.ClipModel(config)
    elif model_arch == ModelSelection.GLAMM:
        return glamm.GlammModel(config)
    elif model_arch == ModelSelection.JANUS:
        return janus.JanusModel(config)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )

    model = get_model(config.architecture, config)
    model.run()
