"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""

import argparse
import logging

from models import llava, qwen
from models.base import ModelBase, ModelSelection


def get_model(model_arch: ModelSelection, model_path: str) -> ModelBase:
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
            return llava.LlavaModel(model_path)
        case ModelSelection.QWEN:
            return qwen.QwenModel(model_path)


if __name__ == '__main__':
    # Initiate parser and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--architecture',
        type=ModelSelection,
        default=ModelSelection.LLAVA,
        choices=list(ModelSelection),
        metavar=f'{[model.value for model in list(ModelSelection)]}',
        help='The model architecture family to extract the embeddings from'
    )
    parser.add_argument(
        '-m',
        '--model-path',
        type=str,
        required=True,
        help='The specific model path to extract the embeddings from'
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='Print out debug statements'
    )

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug(get_model(args.architecture, args.model_path))
