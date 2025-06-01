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
    if model_arch == ModelSelection.LLAVA:
        from models.llava import LlavaModel
        return LlavaModel(config)
    elif model_arch == ModelSelection.QWEN:
        from models.qwen import QwenModel
        return QwenModel(config)
    elif model_arch == ModelSelection.CLIP:
        from models.clip import ClipModel
        return ClipModel(config)
    elif model_arch == ModelSelection.GLAMM:
        from models.glamm import GlammModel
        return GlammModel(config)
    elif model_arch == ModelSelection.JANUS:
        from models.janus import JanusModel
        return JanusModel(config)
    elif model_arch == ModelSelection.BLIP2:
        from models.blip2 import Blip2Model
        return Blip2Model(config)
    elif model_arch == ModelSelection.MOLMO:
        from models.molmo import MolmoModel
        return MolmoModel(config)
    elif model_arch == ModelSelection.INTERNLM_XC:
        from models.internlm_xc import InternLMXComposerModel
        return InternLMXComposerModel(config)
    elif model_arch == ModelSelection.INTERNVL:
        from models.internvl import InternVLModel
        return InternVLModel(config)
    elif model_arch == ModelSelection.MINICPM:
        from models.minicpm import MiniCPMModel
        return MiniCPMModel(config)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )

    model = get_model(config.architecture, config)
    model.run()
