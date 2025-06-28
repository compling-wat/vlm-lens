"""Probe classes for information analysis in models.

Example command: python src/probe/probe.py -c configs/probe.yaml
"""

import argparse
import io
import itertools
import logging
import sqlite3
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
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
            raise ValueError('No GPU found on this machine')

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
            logging.debug(
                'Input database name attribute `db_name` not specified, setting to default `tensors`.')
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

        logging.debug(train_mapping)
        # Set default training config if not provided
        train_mapping.update({k: v for k, v in {
            'optimizer': 'AdamW',
            'learning_rate': 1e-3,
            'loss': 'CrossEntropyLoss',
            'num_epochs': 10,
            'batch_size': 32
        }.items() if k not in train_mapping})

        self.training = train_mapping

        # Load test mapping
        test_mapping = {}
        if hasattr(self, 'test'):
            for mapping in self.test:
                test_mapping = {**test_mapping, **mapping}

        # Set default test config if not provided
        test_mapping.update({k: v for k, v in {
            'optimizer': 'AdamW',
            'learning_rate': 1e-3,
            'loss': 'CrossEntropyLoss',
            'num_epochs': 10,
            'batch_size': 32
        }.items() if k not in test_mapping})

        self.test = test_mapping


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

        self.build_model()

    def build_model(self):
        """Builds the probe model from scratch."""
        # Intialize probe model
        layers = list()
        layers.append(
            nn.Linear(self.config.model['input_size'],
                      self.config.model['hidden_size'])
        )
        layers.append(getattr(nn, self.config.model['activation'])())

        # Intialize intermediate layers based on config
        for _ in range(self.config.model['num_layers'] - 2):
            layers.append(
                nn.Linear(self.config.model['hidden_size'],
                          self.config.model['hidden_size'])
            )
            layers.append(getattr(nn, self.config.model['activation'])())

        # Final layer to output the desired size
        layers.append(
            nn.Linear(self.config.model['hidden_size'],
                      self.config.model['output_size'])
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
        self.config.model.setdefault('output_size', len(all_labels))
        assert (
            'output_size' in self.config.model and len(
                all_labels) == self.config.model['output_size']
        ), 'Input attribute `output_size` does not match number of classes in dataset. Leave blank to assign automatically.'

        # Label to index mapping
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        features, targets = [], []
        probe_layer = self.config.data.get('input_layer', None)
        if not probe_layer:
            logging.debug(
                'No `input_layer` attribute provided for database loading, extracting all tensors...')

        input_size = self.config.data.get('input_size', None)
        for layer, tensor_bytes, label in results:
            if (probe_layer and layer == probe_layer) or (not probe_layer):
                tensor = torch.load(io.BytesIO(tensor_bytes))
                if tensor.ndim > 2:
                    # Apply mean pooling if tensor is not already pooled
                    tensor = tensor.mean(dim=1)
                # Squeeze to shape (hidden_dim)
                tensor = tensor.squeeze()

                if not input_size:
                    # Set model config input_size once
                    input_size = tensor.shape[0]  # pooled tensor
                    self.config.model.setdefault('input_size', input_size)
                    assert (
                        'input_size' in self.config.model and input_size == self.config.model[
                            'input_size']
                    ), 'Input attribute `input_size` does not match input tensor dimension. Leave blank to assign automatically.'

                features.append(tensor)
                targets.append(label_to_idx[label])

        # Stack lists into batched tensors
        X = torch.stack(features)
        Y = torch.tensor(targets)
        logging.debug(f'Features shape {X.shape}, Targets shape {Y.shape}')

        # Move tensors to same device as model
        X, Y = X.to(self.config.device), Y.to(self.config.device)

        return TensorDataset(X, Y)

    def cross_validate(self, config: dict, data: Dataset, nfolds: int = 5) -> float:
        """Trains the model using the config hyperparameters across k folds."""
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)
        val_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(data)))):
            logging.debug(f'===Starting fold {fold}/{nfolds}===')
            train_set, val_set = Subset(data, train_idx), Subset(data, val_idx)
            if fold > 0:
                # Reinitialize model after each fold to prevent contamination
                self.build_model()

            result = self.train(config, train_set, val_set)
            val_losses.append(result['val_loss'])

        # Return lowest validation loss
        return min(val_losses)

    def train(self, train_config: dict, train_set: Dataset, val_set: Dataset = None) -> dict:
        """Train the probe model."""
        logging.debug(
            f'Training the probe model with config {train_config}...')

        # Set the device
        device = torch.device(self.config.device)
        self.model.to(device)

        # Initialize the optimizer
        optimizer_class = getattr(optim, train_config['optimizer'])
        optimizer = optimizer_class(
            self.parameters(), lr=train_config['learning_rate'])

        # Intialize the loss function
        loss_fn = getattr(nn, train_config['loss'])()
        train_loader = DataLoader(
            train_set, batch_size=train_config['batch_size'], shuffle=True)

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

                total_loss += loss.item() * X.size(0)

            mean_train_loss = total_loss / len(train_set)
            logging.debug(
                f"--Epoch {epoch + 1}/{train_config['num_epochs']}: Train loss: {mean_train_loss:.4f}")

        if val_set:
            val_loader = DataLoader(
                val_set, batch_size=train_config['batch_size'])
            # Set model to eval mode and calculate validation loss
            self.model.eval()
            val_loss = 0
            preds, labels = [], []
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    outputs = self.model(X_val)
                    loss = loss_fn(outputs, Y_val)
                    val_loss += loss.item() * X_val.size(0)

                    preds.append(outputs)
                    labels.append(Y_val)

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            val_loss = val_loss / len(val_set)
            val_acc = (preds.argmax(dim=1) == labels).float().mean().item()
            logging.debug(
                f'Validation accuracy: {val_acc}, Validation mean loss: {val_loss}')

            return {'preds': preds, 'labels': labels, 'val_loss': val_loss, 'val_acc': val_acc}

        # TODO: Return train details here
        return {}

    def evaluate(self, test_set: Dataset) -> Tuple[int]:
        """Evaluate the probe model on the input test set."""
        self.model.eval()

        device = torch.device(self.config.device)
        self.model.to(device)

        test_config = self.config.test
        test_loader = DataLoader(
            test_set, batch_size=test_config['batch_size'])

        loss_fn = getattr(nn, test_config['loss'])()
        total_loss = 0.0
        num_correct, num_samples = 0, 0
        with torch.no_grad():
            for X, Y in test_loader:
                outputs = self.model(X)
                loss = loss_fn(outputs, Y)
                total_loss += loss.item() * X.size(0)  # to account for incomplete batches

                preds = outputs.argmax(dim=1)
                num_correct += (preds == Y).sum()
                num_samples += Y.size(0)

        mean_loss = total_loss / len(test_set)
        accuracy = num_correct / num_samples

        logging.debug(
            f'Test accuracy: {accuracy}, Test mean loss: {mean_loss}')
        return accuracy, mean_loss

    def save_model(self) -> None:
        """Saves the trained model to a user-specified path."""
        save_path = self.config.model.get('save_path', 'probe.pth')
        torch.save(self.model.state_dict(), save_path)
        logging.debug(f'Model saved to {save_path}')


def main():
    """Main function to run the probe."""
    config = ProbeConfig()
    probe = Probe(config)

    # Load data and split into train/val and test
    data = probe.data
    indices = list(range(len(data)))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42)
    train_set, test_set = Subset(data, train_idx), Subset(data, test_idx)

    # Load all combinations of hyperparameters
    train_keys = list(config.training.keys())
    train_configs = list(itertools.product(
        *[[config.training[k]] if not isinstance(config.training[k], list) else config.training[k] for k in train_keys]))
    logging.debug(
        f'Hyperparamer tuning using {len(train_configs)} config combinations...')

    # Train using k-fold cross validation on all configs and store the lowest validation losses
    val_losses = []
    for config in train_configs:
        val_loss = probe.cross_validate(
            dict(zip(train_keys, config)), train_set)
        val_losses.append(val_loss)

    # Finally, train the model on the whole train_set using best config
    min_idx = val_losses.index(min(val_losses))
    final_config = dict(zip(train_keys, train_configs[min_idx]))
    probe.train(final_config, train_set)
    logging.debug(
        f'Model train results after hyperparameter tuning: {final_config}')

    # Test the model
    probe.evaluate(test_set)
    probe.save_model()

    # TODO: implement a demo


if __name__ == '__main__':
    main()
