"""main.py.

This module here is the entrypoint to the VLM Lens toolkit.
"""
import logging

from src.models.base import ModelBase
from src.models.config import Config, ModelSelection


def get_model(
    model_arch: ModelSelection,
    config: Config
) -> ModelBase:
    """Returns the model based on the selection enum chosen.

    Args:
        model_arch (ModelSelection): ModelSelection enum chosen for the specific architecture.
        config (Config): The configuration object.

    Returns:
        ModelBase: A model of type ModelBase which implements the runtime
    """
    if model_arch == ModelSelection.LLAVA:
        from src.models.llava import LlavaModel
        return LlavaModel(config)
    elif model_arch == ModelSelection.QWEN:
        from src.models.qwen import QwenModel
        return QwenModel(config)
    elif model_arch == ModelSelection.CLIP:
        from src.models.clip import ClipModel
        return ClipModel(config)
    elif model_arch == ModelSelection.GLAMM:
        from src.models.glamm import GlammModel
        return GlammModel(config)
    elif model_arch == ModelSelection.JANUS:
        from src.models.janus import JanusModel
        return JanusModel(config)
    elif model_arch == ModelSelection.BLIP2:
        from src.models.blip2 import Blip2Model
        return Blip2Model(config)
    elif model_arch == ModelSelection.MOLMO:
        from src.models.molmo import MolmoModel
        return MolmoModel(config)
    elif model_arch == ModelSelection.PALIGEMMA:
        from src.models.paligemma import PaligemmaModel
        return PaligemmaModel(config)
    elif model_arch == ModelSelection.INTERNLM_XC:
        from src.models.internlm_xc import InternLMXComposerModel
        return InternLMXComposerModel(config)
    elif model_arch == ModelSelection.INTERNVL:
        from src.models.internvl import InternVLModel
        return InternVLModel(config)
    elif model_arch == ModelSelection.MINICPM:
        from src.models.minicpm import MiniCPMModel
        return MiniCPMModel(config)
    elif model_arch == ModelSelection.COGVLM:
        from src.models.cogvlm import CogVLMModel
        return CogVLMModel(config)
    elif model_arch == ModelSelection.PIXTRAL:
        from src.models.pixtral import PixtralModel
        return PixtralModel(config)
    elif model_arch == ModelSelection.AYA_VISION:
        from src.models.aya_vision import AyaVisionModel
        return AyaVisionModel(config)
    elif model_arch == ModelSelection.PLM:
        from src.models.plm import PlmModel
        return PlmModel(config)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )

    model = get_model(config.architecture, config)
    model.run()
