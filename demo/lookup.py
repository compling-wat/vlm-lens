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
    LLAVA_15_7B = 'llava-1.5-7b'
    QWENVL_20_2B = 'qwen2-vl-2b-instruct'


# ---- Mapping ----
# model_path: can be a local path or a HF repo id string
# model_spec: absolute Path to the .txt file (we'll return a repo-root-relative string)
_MODEL_MAPPING: dict[ModelVariants, dict[ModelSelection, str, str | Path]] = {
    ModelVariants.LLAVA_15_7B: {
        'model_arch': ModelSelection.LLAVA,
        'model_path': 'llava-hf/llava-1.5-7b-hf',
        'model_spec': SPECS_DIR / 'llava-hf' / 'llava-1.5-7b-hf.txt',
    },
    ModelVariants.QWENVL_20_2B: {
        'model_arch': ModelSelection.QWEN,
        'model_path': 'Qwen/Qwen2-VL-2B-Instruct',
        'model_spec': SPECS_DIR / 'Qwen' / 'Qwen2-VL-2B-Instruct.txt',
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
