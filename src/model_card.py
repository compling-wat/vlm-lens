"""model_card.py.

Probe helper that runs a single forward pass on a loaded model, captures
per-module metadata (class, parameter count, input/output shapes, dtypes), and
writes both a legacy ``logs/<model_path>.txt`` listing and a structured
``docs/_data/cards/<model_path>.json`` artifact consumed by the docs card
renderer.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from src.models.base import ModelBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / 'data' / 'test-images' / 'black_in_blue.png'
DEFAULT_PROMPT = 'Describe this image.'
LOGS_DIR = PROJECT_ROOT / 'logs'
CARDS_DIR = PROJECT_ROOT / 'docs' / '_data' / 'cards'


def _summarize(obj: Any) -> Any:
    """Recursively turn a forward-hook input/output into a JSON-serializable summary.

    Tensor leaves become ``{"shape": [...], "dtype": "..."}``; containers
    preserve their shape; unknown objects record their type name.

    Args:
        obj: Arbitrary forward-hook input or output value.

    Returns:
        A JSON-serializable structure mirroring ``obj``'s shape.
    """
    if isinstance(obj, torch.Tensor):
        return {'shape': list(obj.shape), 'dtype': str(obj.dtype).replace('torch.', '')}
    if isinstance(obj, (list, tuple)):
        return [_summarize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _summarize(v) for k, v in obj.items()}
    # transformers ModelOutput and similar dataclass-style containers
    to_tuple = getattr(obj, 'to_tuple', None)
    if callable(to_tuple):
        try:
            return [_summarize(x) for x in to_tuple()]
        except Exception:
            pass
    if obj is None:
        return None
    return {'type': type(obj).__name__}


def _total_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _prepare_fixture(model: 'ModelBase') -> tuple[str, str, Any]:
    """Build the probe fixture for ``model``.

    Populates the minimal Config fields the model's forward pipeline needs and
    generates a processor output.

    Args:
        model: Loaded model wrapper whose ``config`` will be mutated in place.

    Returns:
        Tuple ``(image_path, prompt, processor_output)``.
    """
    from src.models.config import Config  # noqa: F401 (import side effects)

    image_path = str(DEFAULT_IMAGE)
    prompt = DEFAULT_PROMPT

    # Minimally populate fields that _generate_prompt / _generate_processor_output
    # / _forward expect — Config's early-return on -l skips these normally.
    # Force dataset=None unconditionally: YAMLs that declare `dataset:` leave
    # it as a raw list pre-parse, which would break Config.has_images().
    cfg = model.config
    cfg.dataset = None
    cfg.image_paths = [image_path]
    if not hasattr(cfg, 'prompt') or not isinstance(cfg.prompt, str):
        cfg.prompt = prompt
    if not hasattr(cfg, 'NO_IMG_PROMPT'):
        cfg.NO_IMG_PROMPT = 'No image prompt'
    if not hasattr(cfg, 'DB_TABLE_NAME'):
        cfg.DB_TABLE_NAME = 'tensors'
    if not hasattr(cfg, 'device') or not isinstance(cfg.device, torch.device):
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chat_prompt = model._generate_prompt(prompt)
    data = model._generate_processor_output(prompt=chat_prompt, img_path=image_path)
    # Processor output is FP32 by default, but the probe forces
    # `torch_dtype='auto'` which resolves to BF16/FP16 for many checkpoints.
    # Cast floating-point inputs to match the model's dtype so the fixture
    # forward pass actually runs (otherwise Conv2d/MatMul raises a dtype
    # mismatch and we end up with partial shapes in the card).
    target_dtype = next(
        (p.dtype for p in model.model.parameters() if p.is_floating_point()),
        None,
    )
    if target_dtype is not None and target_dtype != torch.float32:
        data = _cast_floats(data, target_dtype)
    return image_path, prompt, data


def _cast_floats(obj: Any, dtype: torch.dtype) -> Any:
    """Recursively cast FP32 tensors in a container to ``dtype``.

    Handles ``BatchFeature`` (``UserDict``-derived) as well as plain
    dicts/lists/tuples. Integer tensors are left alone.

    Args:
        obj: Container or tensor to cast.
        dtype: Target floating-point dtype.

    Returns:
        A structure mirroring ``obj`` with floating-point tensors cast to
        ``dtype``. ``Mapping`` subclasses are mutated in place when possible.
    """
    from collections.abc import Mapping
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, list):
        return [_cast_floats(x, dtype) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_floats(x, dtype) for x in obj)
    if isinstance(obj, Mapping):
        # Mutate in place for ``BatchFeature``/``UserDict`` — rebuilding a
        # plain dict would strip the container's downstream-expected type.
        try:
            for k in list(obj.keys()):
                obj[k] = _cast_floats(obj[k], dtype)
            return obj
        except Exception:
            return {k: _cast_floats(v, dtype) for k, v in obj.items()}
    return obj


def _write_logs_txt(model: 'ModelBase') -> Path:
    """Preserve the legacy logs/<model_path>.txt listing (soft-deprecation window).

    Args:
        model: Loaded model wrapper whose modules are enumerated.

    Returns:
        Path to the written ``.txt`` file.
    """
    out = LOGS_DIR / f'{model.model_path}.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for name, _ in model.model.named_modules():
            f.write(f'{name}\n')
    return out


def run_probe(model: 'ModelBase') -> Path:
    """Run one forward pass, capture per-module shape metadata, write card JSON.

    Args:
        model: Loaded model wrapper to probe.

    Returns:
        Path to the written card JSON file.
    """
    logs_path = _write_logs_txt(model)
    logging.info(f'Wrote legacy module listing to {logs_path}')

    image_path, prompt, data = _prepare_fixture(model)

    records: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}

    for name, module in model.model.named_modules():
        rec = {
            'name': name,
            'class': type(module).__name__,
            'params_direct': sum(p.numel() for p in module.parameters(recurse=False)),
            'params_total': _total_params(module),
            'input_shapes': None,
            'output_shapes': None,
            'called': False,
        }
        records.append(rec)
        by_name[name] = rec

    def make_hook(rec: dict[str, Any]) -> Any:
        def hook(_module: torch.nn.Module, inputs: tuple, output: Any) -> None:
            if rec['called']:
                return
            rec['called'] = True
            rec['input_shapes'] = _summarize(inputs)
            rec['output_shapes'] = _summarize(output)
        return hook

    handles = []
    for name, module in model.model.named_modules():
        handles.append(module.register_forward_hook(make_hook(by_name[name])))

    model.model.to(model.config.device)
    model.model.eval()

    try:
        with torch.no_grad():
            model._forward(data)
    except Exception as e:  # noqa: BLE001
        logging.warning(
            f'Forward pass raised {type(e).__name__}: {e}. Recording partial shapes.'
        )
    finally:
        for h in handles:
            h.remove()

    card = {
        'architecture': model.config.architecture.value,
        'model_path': model.model_path,
        'total_params': _total_params(model.model),
        'fixture': {
            'image': os.path.relpath(image_path, PROJECT_ROOT),
            'prompt': prompt,
        },
        'device': str(model.config.device),
        'modules': records,
    }

    out_path = CARDS_DIR / f'{model.model_path}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(card, f, indent=2)
    logging.info(f'Wrote model card to {out_path}')
    return out_path
