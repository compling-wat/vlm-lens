"""read_tensor.py.

Outputs the layers and its specific associated tensors.
"""

import logging
import os
import pickle
import sqlite3
import sys
from typing import List, Tuple

import torch


def retrieve_tensors(
    config,
    layer: str,
    query_img_path: List[str]
) -> List[Tuple[str, torch.Tensor]]:
    """Retrieve a PyTorch tensor based on its inputs and config.

    Args:
        config (Config): The path to the configuration itself.
        layer (str): The name of the layer itself.
        query_img_path (str): Image path to query for.

    Returns:
        List[Tuple[str, torch.Tensor]]:
            List of tuple of the layer and its retrieved tensor.
    """
    connection = sqlite3.connect(config.output_db)
    cursor = connection.cursor()

    # Build the query
    query = f"""
        SELECT layer, tensor, timestamp, image_path, prompt FROM {config.DB_TABLE_NAME}
    """
    conditions = []
    params = []

    args = {
        'name': config.model_path,
        'architecture': config.architecture.value,
        'prompt': config.prompt,
        'image_path': query_img_path,
        'layer': layer
    }

    for key, value in args.items():
        if value is None:
            continue

        conditions.append(f'{key} = ?')
        params.append(value)

    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)

    logging.debug(f'Query: {query}')

    # Execute the query
    cursor.execute(query, params)

    # Fetch the results
    results = cursor.fetchall()

    # Close the connection
    connection.close()

    # Convert the binary blobs back to tensors
    tensors = []
    for result in results:
        layer, tensor, timestamp, image_path, prompt = result
        tensor = pickle.loads(tensor)
        tensors.append((layer, tensor, timestamp, image_path, prompt))

    return tensors


def get_unique_layers(config):
    """Retrieve a all unique layers states saved based on its config.

    Args:
        config (Config): The path to the configuration itself.
    """
    # Connect to the database
    connection = sqlite3.connect(config.output_db)
    cursor = connection.cursor()

    query = f"""
        SELECT DISTINCT layer FROM {config.DB_TABLE_NAME}
    """

    conditions = []
    params = []

    args = {
        'name': config.model_path,
        'architecture': config.architecture.value
    }

    for key, value in args.items():
        if value is None:
            continue

        conditions.append(f'{key} = ?')
        params.append(value)

    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
        query += ';'

    logging.debug(f'Query: {query}')
    logging.debug(f'Params: {params}')

    # Execute the query
    cursor.execute(query, params)

    # Fetch the results
    results = cursor.fetchall()

    # Close the connection
    connection.close()

    layers = [result[0] for result in results]

    return layers


if __name__ == '__main__':
    # first we want to add the current directory into the path
    EXC_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(EXC_DIR)
    from src.models.config import Config

    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )

    unique_layers = get_unique_layers(config)
    print(f'Unique layers: {unique_layers}')

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
        print(f'~~Tensors for {query_img_path}~~')
        for layer in unique_layers:
            if not config.matches_module(layer):
                continue
            tensors = retrieve_tensors(config, layer, query_img_path)
            for layer, tensor, timestamp, image_path, prompt in tensors:
                print(
                    f'Name: {config.model_path}, '
                    f'Architecture: {config.architecture.value}, '
                    f'Layer: {layer}, '
                    f'Tensor Norm: {torch.norm(tensor)}, '
                    f'Timestamp: {timestamp}, Image path: {image_path}, '
                    f'Prompt: {prompt}'
                )
        print()
