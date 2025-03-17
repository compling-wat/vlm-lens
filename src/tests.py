"""Unit tests for the model classes."""
import logging

import torch

from main import get_model
# NOTE: import your model
from models.config import Config

# input ids are integers

# hidden states are tensors of floats (by torch loading)

# input the same data, get the same output

# input different data, get different output

# input images of different sizes, get the same output shape

# input ids dimension in batch = batch size * input ids as single input

# work on image only inputs (optional)

# work on text only inputs (optional)


if __name__ == '__main__':
    config = Config()
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    model = get_model(config.architecture, config)
    input_data = model.load_input_data()
    model.forward(input_data)
    model.save_states()
    states = torch.load(model.config.output_dir)

    # tests
    pass
