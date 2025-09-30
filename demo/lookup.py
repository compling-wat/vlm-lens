"""Model info lookup utilities."""

import os
from enum import Enum
from pathlib import Path
from typing import Tuple

from src.models.config import ModelSelection

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS_DIR = Path(os.getenv('MODEL_SPECS_DIR', REPO_ROOT / 'logs'))

# TODO: To store local model weights in the repo, also define:
# MODELS_DIR = Path(os.getenv('MODELS_DIR', REPO_ROOT / 'checkpoints'))


class ModelVariants(str, Enum):
    """Enum that contains all possible model variants."""
    AYA_VISION_8B = 'aya-vision-8b'
    BLIP2_3B = 'blip2-opt-2.7b'
    COGVLM_17B = 'cogvlm-17b'
    GLAMM_7B = 'glamm-7b'
    INTERNLM_XC_25_7B = 'internlm-xcomposer2.5-7b'
    INTERNVL_25_8B = 'internvl-2.5-8b'
    JANUS_1B = 'janus-pro-1b'
    LLAVA_15_7B = 'llava-1.5-7b'
    MINICPM_O_26_8B = 'minicpm-o-2.6-8b'
    MINICPM_V_20_3B = 'minicpm-v-2.0-2.8b'
    MOLMO_7B = 'molmo-7b'
    PALIGEMMA_3B = 'paligemma-3b'
    PIXTRAL_12B = 'pixtral-12b'
    PERCEPTION_LM_1B = 'perception-lm-1b'
    QWENVL_20_2B = 'qwen2-vl-2b-instruct'
    QWENVL_20_7B = 'qwen2-vl-7b-instruct'
    # TODO: Add more models here as needed.


# ---- Mapping ----
# model_path: can be a local path or a HF repo id string
# model_spec: absolute Path to the .txt file (we'll return a repo-root-relative string)
_MODEL_MAPPING: dict[ModelVariants, dict[ModelSelection, str, str | Path]] = {
    ModelVariants.AYA_VISION_8B: {
        'model_arch': ModelSelection.AYA_VISION,
        'model_path': 'CohereLabs/aya-vision-8b',
        'model_spec': SPECS_DIR / 'CohereLabs' / 'aya-vision-8b.txt',
    },
    ModelVariants.BLIP2_3B: {
        'model_arch': ModelSelection.BLIP2,
        'model_path': 'Salesforce/blip2-opt-2.7b',
        'model_spec': SPECS_DIR / 'Salesforce' / 'blip2-opt-2.7b.txt',
    },
    ModelVariants.COGVLM_17B: {
        'model_arch': ModelSelection.COGVLM,
        'model_path': 'THUDM/cogvlm-chat-hf',
        'model_spec': SPECS_DIR / 'THUDM' / 'cogvlm-chat-hf.txt',
    },
    ModelVariants.GLAMM_7B: {
        'model_arch': ModelSelection.GLAMM,
        'model_path': 'MBZUAI/GLaMM-FullScope',
        'model_spec': SPECS_DIR / 'MBZUAI' / 'GLaMM-FullScope.txt',
    },
    ModelVariants.INTERNLM_XC_25_7B: {
        'model_arch': ModelSelection.INTERNLM_XC,
        'model_path': 'internlm/internlm-xcomposer2d5-7b',
        'model_spec': SPECS_DIR / 'internlm' / 'internlm-xcomposer2d5-7b.txt',
    },
    ModelVariants.INTERNVL_25_8B: {
        'model_arch': ModelSelection.INTERNVL,
        'model_path': 'OpenGVLab/InternVL2_5-8B',
        'model_spec': SPECS_DIR / 'internvl' / 'InternVL2_5-8B.txt',
    },
    ModelVariants.JANUS_1B: {
        'model_arch': ModelSelection.JANUS,
        'model_path': 'deepseek-community/Janus-Pro-1B',
        'model_spec': SPECS_DIR / 'deepseek-community' / 'Janus-Pro-1B.txt',
    },
    ModelVariants.LLAVA_15_7B: {
        'model_arch': ModelSelection.LLAVA,
        'model_path': 'llava-hf/llava-1.5-7b-hf',
        'model_spec': SPECS_DIR / 'llava-hf' / 'llava-1.5-7b-hf.txt',
    },
    ModelVariants.MINICPM_O_26_8B: {
        'model_arch': ModelSelection.MINICPM,
        'model_path': 'openbmb/MiniCPM-o-2_6',
        'model_spec': SPECS_DIR / 'openbmb' / 'MiniCPM-o-2_6.txt',
    },
    ModelVariants.MINICPM_V_20_3B: {
        'model_arch': ModelSelection.MINICPM,
        'model_path': 'compling/MiniCPM-V-2',
        'model_spec': SPECS_DIR / 'wonderwind271' / 'MiniCPM-V-2.txt',
    },
    ModelVariants.MOLMO_7B: {
        'model_arch': ModelSelection.MOLMO,
        'model_path': 'allenai/Molmo-7B-D-0924',
        'model_spec': SPECS_DIR / 'allenai' / 'Molmo-7B-D-0924.txt',
    },
    ModelVariants.PALIGEMMA_3B: {
        'model_arch': ModelSelection.PALIGEMMA,
        'model_path': 'google/paligemma-3b-mix-224',
        'model_spec': SPECS_DIR / 'paligemma' / 'paligemma-3b.txt',
    },
    ModelVariants.PIXTRAL_12B: {
        'model_arch': ModelSelection.PIXTRAL,
        'model_path': 'mistralai/Pixtral-12B-2409',
        'model_spec': SPECS_DIR / 'mistralai' / 'Pixtral-12B-2409.txt',
    },
    ModelVariants.PERCEPTION_LM_1B: {
        'model_arch': ModelSelection.PLM,
        'model_path': 'facebook/Perception-LM-1B',
        'model_spec': SPECS_DIR / 'facebook' / 'Perception-LM-1B.txt',
    },
    ModelVariants.QWENVL_20_2B: {
        'model_arch': ModelSelection.QWEN,
        'model_path': 'Qwen/Qwen2-VL-2B-Instruct',
        'model_spec': SPECS_DIR / 'Qwen' / 'Qwen2-VL-2B-Instruct.txt',
    },
    ModelVariants.QWENVL_20_7B: {
        'model_arch': ModelSelection.QWEN,
        'model_path': 'Qwen/Qwen2-VL-7B-Instruct',
        'model_spec': SPECS_DIR / 'Qwen' / 'Qwen2-VL-7B-Instruct.txt',
    },
    # TODO: Add more models here as needed.
}


def _to_repo_relative(p: Path) -> str:
    """Convert a path to a repo-root–relative string if possible.

    Args:
        p (Path): The path to convert.

    Returns:
        str: `p` relative to ``REPO_ROOT`` if `p` is within it; otherwise the
            absolute path as a string.
    """
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def get_model_info(model_var: ModelVariants) -> Tuple[ModelSelection, str, str]:
    """Return the model path and spec link for a given selection.

    Args:
        model_var (ModelVariants): The model variant to look up.

    Returns:
        Tuple[ModelSelection, str, str]:
            A triple of ``(model_selection, model_path, link_to_model_spec)`` where
            `model_selection` is a ModelSelection enum entry,
            `model_path` is an HF repo id or local path, and
            `link_to_model_spec` is a repo-root-relative path to the spec ``.txt``.

    Raises:
        KeyError: If the provided `model` is unknown / not in the mapping.
        FileNotFoundError: If the resolved spec file does not exist.
    """
    try:
        info = _MODEL_MAPPING[model_var]
    except KeyError as e:
        raise KeyError(f'Unknown model: {model_var!r}') from e

    model_selection = ModelSelection(info['model_arch'])
    model_path = str(info['model_path'])
    spec_path = Path(info['model_spec']).resolve()

    if not spec_path.exists():
        raise FileNotFoundError(f'Spec file not found: {spec_path}')

    return model_selection, model_path, _to_repo_relative(spec_path)
