"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""

import argparse
import logging

from models import llava, qwen
from models.base import ModelBase, ModelSelection


def get_model(model_sel: ModelSelection) -> ModelBase:
    """Returns the model based on the selection enum chosen.

    Args:
        model_sel (ModelSelection): ModelSelection enum chosen

    Returns:
        base.ModelBase: A model of type ModelBase which implements the runtime
    """
    match model_sel:
        case ModelSelection.LLAVA:
            return llava.LlavaModel()
        case ModelSelection.QWEN:
            return qwen.QwenModel()
        case ModelSelection.QWEN_2B:
            return qwen.QwenModel_2B()


if __name__ == '__main__':
    # Initiate parser and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=ModelSelection,
        default=ModelSelection.LLAVA,
        choices=list(ModelSelection),
        metavar=f'{[model.value for model in list(ModelSelection)]}',
        help='The model to extract the embeddings from'
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='The model to extract the embeddings from'
    )

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug(get_model(args.model))
