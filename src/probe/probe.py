"""Probe classes for information analysis in models.

Example command: python src/probe/probe.py -c configs/probe.yaml
"""
import argparse
import torch.nn as nn
import yaml

from typing import Dict, Any


class Config():
    """Configuration class for the probe."""

    def __init__(self):
        """Initialize the configuration."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-c',
            '--config',
            type=str,
            help='Path to the probe configuration file'
        )
        args = parser.parse_args()

        assert args.config is not None, 'Config file must be provided.'
        with open(args.config, 'r') as file:
            data = yaml.safe_load(file)
            for key in data.keys():
                setattr(self, key, data[key])


class Probe(nn.Module):
    """Probe class for extracting information from models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super(Probe, self).__init__()
        layers = list()
        layers.append(nn.Linear(config.input_size, config.hidden_size))
        layers.append(getattr(nn, config.activation)())
        for _ in range(config.num_layers - 2):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            layers.append(getattr(nn, config.activation)())
        layers.append(nn.Linear(config.hidden_size, config.output_size))
        self.model = nn.Sequential(layers)


    def forward(self, x):
        """Forward pass of the probe."""
        return self.model(x)


# TODO: design and implement the probe main function
# TODO: support all configs shown in the yaml with the probe config
# TODO: implement a demo

if __name__ == '__main__':
    config = Config()
    probe = Probe(config)
