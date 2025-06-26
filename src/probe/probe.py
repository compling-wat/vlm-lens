"""Probe classes for information analysis in models.

Example command: python src/probe/probe.py -c configs/probe.yaml
"""

import argparse
import io
import logging
import sqlite3
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


class ProbeConfig:
    """Configuration class for the probe."""

    def __init__(self):
        """Initialize the configuration."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-c', '--config', type=str, help='Path to the probe configuration file'
        )

        parser.add_argument(
            '--debug',
            default=False,
            action='store_true',
            help='Flag to print out debug statements',
        )

        parser.add_argument(
            '-d',
            '--device',
            type=str,
            default='cpu',
            help='The device to send the model and tensors to',
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

        # Load data mapping
        assert (
            hasattr(self, 'data')
        ), 'The `data` field must be specified in the config, with an input database path.'

        data_mapping = {}
        for mapping in self.data:
            data_mapping = {**data_mapping, **mapping}

        # Check if specific layer in specified for the database
        data_mapping.setdefault('input_layer', None)

        # Set default database name if not specified
        if 'db_name' not in data_mapping:
            logging.debug('Input database name attribute `db_name` not specified, setting to default `tensors`.')
            data_mapping.setdefault('db_name', 'tensors')
        self.data = data_mapping

        # Load model mapping
        model_mapping = {}
        if hasattr(self, 'model'):
            for mapping in self.model:
                model_mapping = {**model_mapping, **mapping}

        # Set default model config if not provided
        # input_size and output_size will be set when the data is loaded
        model_mapping.update({k: v for k, v in {
                        'activation': 'ReLU',
                        'hidden_size': 256,
                        'num_layers': 2,
                    }.items() if k not in model_mapping})
        logging.debug(model_mapping)
        self.model = model_mapping

        # Load training mapping
        train_mapping = {}
        if hasattr(self, 'training'):
            for mapping in self.training:
                train_mapping = {**train_mapping, **mapping}

        # Set default training config if not provided
        train_mapping.update({k: v for k, v in {
                        'optimizer': 'AdamW',
                        'learning_rate': 1e-3,
                        'loss': 'MSELoss',
                        'num_epochs': 10,
                        'batch_size': 32
                    }.items() if k not in train_mapping})
        self.training = train_mapping


class Probe(nn.Module):
    """Probe class for extracting information from models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Intialize the probe with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the probe.
        """
        super(Probe, self).__init__()
        self.config = config

        # Load input data to parse model input_size and output_size
        self.data = self.load_data()

        # Intialize probe model
        layers = list()
        layers.append(
            nn.Linear(config.model['input_size'], config.model['hidden_size'])
        )
        layers.append(getattr(nn, config.model['activation'])())

        # Intialize intermediate layers based on config
        for _ in range(config.model['num_layers'] - 2):
            layers.append(
                nn.Linear(config.model['hidden_size'], config.model['hidden_size'])
            )
            layers.append(getattr(nn, config.model['activation'])())

        # Final layer to output the desired size
        layers.append(
            nn.Linear(config.model['hidden_size'], config.model['output_size'])
        )

        # Combine all layers to construct the model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the probe model."""
        logging.debug('Forward pass with input: %s', x.shape)
        return self.model(x)

    def load_data(self) -> TensorDataset:
        """Load tensors from the database."""
        logging.debug('Loading tensors from the database...')
        # Connect to database
        connection = sqlite3.connect(self.config.data['input_db'])
        cursor = connection.cursor()

        # Build query and fetch results
        cursor.execute(
            f"SELECT layer, tensor, label FROM {self.config.data['db_name']}"
        )
        results = cursor.fetchall()

        # Close the connection
        connection.close()

        # Gather unique class labels
        all_labels = set([result[2] for result in results])
        logging.debug(len(all_labels))
        self.config.model.setdefault('output_size', len(all_labels))
        assert (
            'output_size' in self.config.model and len(all_labels) == self.config.model['output_size']
        ), 'Input attribute `output_size` does not match number of classes in dataset. Leave blank to assign automatically.'

        # Label to one-hot tensor mapping
        label_to_tensor = {
            label: F.one_hot(torch.tensor(i), num_classes=len(all_labels)).float()
            for i, label in enumerate(all_labels)
        }

        features, targets = [], []
        probe_layer = self.config.data.get('input_layer', None)
        if not probe_layer:
            logging.debug('No `input_layer` attribute provided for database loading, extracting all tensors...')

        input_size = self.config.data.get('input_size', None)
        for layer, tensor_bytes, label in results:
            if (probe_layer and layer == probe_layer) or (not probe_layer):
                tensor = torch.load(io.BytesIO(tensor_bytes))
                if tensor.ndim > 2:
                    # Apply mean pooling if tensor is already pooled
                    tensor = tensor.mean(dim=1)
                # Squeeze to shape (hidden_dim)
                tensor = tensor.squeeze()

                if not input_size:
                    # Set model config input_size once
                    input_size = tensor.shape[0]  # pooled tensor
                    self.config.model.setdefault('input_size', input_size)
                    assert (
                        'input_size' in self.config.model and input_size == self.config.model['input_size']
                        ), 'Input attribute `input_size` does not match input tensor dimension. Leave blank to assign automatically.'

                features.append(tensor)
                targets.append(label_to_tensor[label])

        # Stack lists into batched tensors
        X, Y = torch.stack(features), torch.stack(targets)
        logging.debug(f'X.shape {X.shape} Y.shape {Y.shape}')
        # Move tensors to same device as model
        X, Y = X.to(self.config.device), Y.to(self.config.device)

        return TensorDataset(X, Y)

    def train(self, data: Dataset, kfold: int = 5) -> None:
        """Train the probe model."""
        logging.debug('Training the probe model...')
        train_config = self.config.training

        # Set the device
        device = torch.device(self.config.device)
        self.to(device)

        # Initialize the optimizer
        optimizer_class = getattr(optim, train_config['optimizer'])
        optimizer = optimizer_class(self.parameters(), lr=train_config['learning_rate'])

        # Intialize the loss function
        loss_fn = getattr(nn, train_config['loss'])()

        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        # TODO: test this by passing Dataset and not Subset
        kf_split = kf.split(range(len(data)))
        for fold, (train_idx, val_idx) in enumerate(kf_split):
            logging.debug(f'===Starting fold {fold}/{kfold}===')
            train_set, val_set = Subset(data, train_idx), Subset(data, val_idx)

            train_loader = DataLoader(
                train_set, batch_size=train_config['batch_size'], shuffle=True
            )
            val_loader = DataLoader(val_set, batch_size=train_config['batch_size'])

            for epoch in range(train_config['num_epochs']):
                # Set the model to training mode
                self.model.train()
                total_loss = 0
                for X, Y in train_loader:
                    optimizer.zero_grad()

                    outputs = self.model(X)
                    loss = loss_fn(outputs, Y)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                mean_train_loss = total_loss / len(train_loader)

                # Set model to eval mode and calculate validation loss
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_val, Y_val in val_loader:
                        outputs = self.model(X_val)
                        loss = loss_fn(outputs, Y_val)
                        val_loss += loss.item()

                mean_val_loss = val_loss / len(val_loader)

                logging.debug(f"--Epoch {epoch + 1}/{train_config['num_epochs']}: Train loss: {mean_train_loss:.4f}, Validation loss: {mean_val_loss:.4f}")

        self.save_model()
        return

    def save_model(self) -> None:
        """Saves the trained model to a user-specified path."""
        save_path = self.config.model.get('save_path', 'probe.pth')
        torch.save(self.model.state_dict(), save_path)
        logging.debug(f'Model saved to {save_path}')


def main():
    """Main function to run the probe."""
    config = ProbeConfig()

    logging.debug('Initializing Probe with config: %s', config)
    probe = Probe(config)

    # Load data and split into train/val and test
    data = probe.data
    indices = list(range(len(data)))

    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = Subset(data, train_idx)
    # test_set = Subset(data, test_idx)

    probe.train(train_data)

    # TODO: implement a demo


if __name__ == '__main__':
    main()
