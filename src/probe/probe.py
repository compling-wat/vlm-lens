"""Probe classes for information analysis in models.

Example command: python src/probe/probe.py -c configs/probe.yaml
"""
import argparse
import io
import logging
import sqlite3
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml


class ProbeConfig():
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

        parser.add_argument(
            '-d',
            '--debug',
            default=False,
            action='store_true',
            help='Flag to print out debug statements'
        )

        parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='The device to send the model and tensors to'
        )

        args = parser.parse_args()

        assert args.config is not None, 'Config file must be provided.'
        with open(args.config, 'r') as file:
            data = yaml.safe_load(file)
            for key in data.keys():
                setattr(self, key, data[key])

        # Set debug mode based on config
        logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

        # Load model device
        if 'cuda' in args.device and not torch.cuda.is_available():
            raise ValueError('No GPU found for this machine')

        self.device = args.device
        logging.debug(self.device)

        # Load input database
        assert hasattr(self, 'data') and 'input_db' in self.data, (
            'Input database must be specified in the configuration.'
        )

        # Check if output database is specified, use probe_output.db as default
        if 'output_db' not in self.data:
            self.data['output_db'] = 'probe_output.db'

        # Set default database name if not specified
        if 'db_name' not in self.data:
            logging.debug('Database name not specified, setting to default "tensors".')
            self.data['db_name'] = 'tensors'

        # Intialize the model config attributes
        if not hasattr(self, 'model'):
            # TODO: figure out how to automatically set input/hidden/output size
            self.model = {
                'activation': 'ReLU',
                'num_layers': 2
            }

        # Intialize the training config attributes
        if not hasattr(self, 'training'):
            self.training = {}


class Probe(nn.Module):
    """Probe class for extracting information from models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Intialize the probe with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the probe.
        """
        super(Probe, self).__init__()
        layers = list()
        self.config = config
        model_config = config.model

        # Intialize the input layer and activation
        layers.append(nn.Linear(model_config['input_size'], model_config['hidden_size']))
        layers.append(getattr(nn, model_config['activation'])())

        # Intialize intermediate layers based on config
        for _ in range(model_config['num_layers'] - 2):
            layers.append(nn.Linear(model_config['hidden_size'], model_config['hidden_size']))
            layers.append(getattr(nn, model_config['activation'])())

        # Final layer to output the desired size
        layers.append(nn.Linear(model_config['hidden_size'], model_config['output_size']))

        # Combine all layers to construct the model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the probe model."""
        logging.debug('Forward pass with input: %s', x.shape)
        return self.model(x)

    def load_data(self) -> Tuple[torch.Tensor]:
        """Load tensors from the database."""
        logging.debug('Loading tensors from the database...')
        # Connect to database
        connection = sqlite3.connect(self.config.data['input_db'])
        cursor = connection.cursor()

        # Build query and fetch results
        cursor.execute(f"SELECT layer, tensor, answer FROM {self.config.data['db_name']}")
        results = cursor.fetchall()

        # Close the connection
        connection.close()

        # Gather unique class labels
        all_labels = [result[2] for result in results]
        assert len(all_labels) == self.config.model['output_size'], (
            'Input number of classes does not match dataset classes.'
        )

        # Label to one-hot tensor mapping
        label_to_tensor = {label: F.one_hot(
                                torch.tensor(i),
                                num_classes=len(all_labels)
                                ).float() for i, label in enumerate(all_labels)}

        # Construct feature tensors
        features = []
        targets = []
        for _, tensor_bytes, answer in results:
            # Mean pool the input tensor to shape (hidden_size)
            tensor = torch.load(io.BytesIO(tensor_bytes)).last_hidden_state
            pooled_tensor = tensor.mean(dim=1).squeeze(0)
            features.append(pooled_tensor)

            # Add one-hot label of entry
            targets.append(label_to_tensor[answer])

        # Stack lists into batched tensors
        X, Y = torch.stack(features), torch.stack(targets)

        # Move tensors to same device as model
        X, Y = X.to(self.config.device), Y.to(self.config.device)

        return X, Y

    def train(self, data: Tuple[torch.Tensor]) -> None:
        """Train the probe model.

        Args:
            data (Tuple[torch.Tensor]]): Tuple of feature and target tensors to train the probe.
        """
        logging.debug('Training the probe model...')
        train_config = self.config.training
        X, Y = data

        # Set attributes if not specified
        train_config.setdefault('optimizer', 'AdamW')
        train_config.setdefault('learning_rate', 1e-3)
        train_config.setdefault('loss', 'MSELoss')
        train_config.setdefault('num_epochs', 10)
        train_config.setdefault('batch_size', 32)

        logging.debug('Training configuration: %s', train_config)

        # Set the model to training mode
        self.model.train()

        # Set the device
        device = torch.device(self.config.device)
        self.to(device)

        # Initialize the optimizer
        optimizer_class = getattr(optim, train_config['optimizer'])
        optimizer = optimizer_class(self.parameters(), lr=train_config['learning_rate'])

        # Intialize the loss function
        loss_fn = getattr(nn, train_config['loss'])()

        for epoch in range(train_config['num_epochs']):
            logging.debug(f'Starting epoch {epoch + 1}/{train_config["num_epochs"]}')

            optimizer.zero_grad()
            outputs = self.model(X)

            loss = loss_fn(outputs, Y)
            logging.debug(f'loss {loss}')
            loss.backward()
            optimizer.step()

        return


def main():
    """Main function to run the probe."""
    config = ProbeConfig()

    logging.debug('Initializing Probe with config: %s', config)
    probe = Probe(config)

    data = probe.load_data()

    probe.train(data)

    # TODO: support all configs shown in the yaml with the probe config
    # TODO: implement a demo


if __name__ == '__main__':
    main()
