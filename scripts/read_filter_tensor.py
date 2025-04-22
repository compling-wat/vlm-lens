"""read_filter_tensor.py.

Outputs the layers and its specific associated tensors according to the filters
defined.
"""
import logging
import os
import sys

import torch
from read_tensor import get_unique_layers, retrieve_tensors

if __name__ == '__main__':
    # first we want to add the current directory into the path
    EXC_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(EXC_DIR)
    sys.path.append(os.path.join(EXC_DIR, 'src/'))
    from src.models.config import Config
    from src.run_filters import FilterConfig

    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )
    filter_config = FilterConfig()
    logging.debug(f'Filter config is set to {filter_config}')

    unique_layers = get_unique_layers(config)
    logging.info(f'Unique layers: {unique_layers}')

    for key, data in filter_config.filter_data.items():
        logging.info(f'Data from filter {key}')

        if 'input_dir' in data.keys():
            config.set_image_paths(data['input_dir'])
        else:
            config.set_image_paths(config.default_input_dir)
        if 'modules' in data.keys():
            config.set_modules(data['modules'])
        else:
            config.set_modules(config.default_modules)
        if 'prompt' in data.keys():
            config.set_prompt(data['prompt'])
        else:
            config.set_prompt(config.default_prompt)

        image_paths = (
            [config.NO_IMG_PROMPT]
            if len(config.image_paths) == 0 else
            config.image_paths
        )

        for query_img_path in image_paths:
            query_img_path = (
                os.path.abspath(query_img_path)
                if query_img_path != config.NO_IMG_PROMPT else
                query_img_path
            )
            logging.info(f'~~Tensors for {query_img_path}~~')
            for layer in unique_layers:
                if not config.matches_module(layer):
                    continue
                tensors = retrieve_tensors(config, layer, query_img_path)
                for layer, tensor, timestamp, image_path, prompt in tensors:
                    logging.info(
                        f'Name: {config.model_path}, '
                        f'Architecture: {config.architecture.value}, '
                        f'Layer: {layer}, '
                        f'Tensor Norm: {torch.norm(tensor)}, '
                        f'Timestamp: {timestamp}, Image path: {image_path}, '
                        f'Prompt: {prompt}'
                    )
