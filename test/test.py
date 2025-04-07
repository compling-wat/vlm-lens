"""test/test.py.

This module tests the functionality of hidden state extraction model classes.
"""
import importlib
import os
import sys
from typing import List, Tuple

import pytest
import torch


# TODO: add your config file here
@pytest.mark.parametrize(
    ('config_path'),
    [
        'configs/qwen-2b.yaml',
        # 'configs/clip-base.yaml',
    ]
)
class TestHiddenStates:
    """test class for hidden states extraction functionality."""
    def _import(self):
        """Initialize the test cases."""
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self._get_unique_layers = importlib.import_module('scripts.read_tensor').get_unique_layers
        self._retrieve_tensors = importlib.import_module('scripts.read_tensor').retrieve_tensors
        self.ModelBase = importlib.import_module('src.models.base').ModelBase
        self.Config = importlib.import_module('src.models.config').Config
        self.ModelSelection = importlib.import_module('src.models.config').ModelSelection

    # TODO: add your model imports here
    def _get_test_model(self, model_arch, config):
        """Create a test model instance."""
        if model_arch == self.ModelSelection.LLAVA:
            from src.models import llava
            return llava.LlavaModel(config)
        elif model_arch == self.ModelSelection.QWEN:
            from src.models import qwen
            return qwen.QwenModel(config)
        elif model_arch == self.ModelSelection.CLIP:
            from src.models import clip
            return clip.ClipModel(config)
        else:
            raise ValueError(f'Unknown model architecture: {model_arch}')

    def _model_run(self,
                   config_path: str
                   ) -> List[Tuple[str, Tuple[torch.Tensor], str, str, str]]:
        """Initialize the model and run it."""
        # get everything from the config
        sys.argv = ['test/test.py', '-c', config_path, '--debug']
        config = self.Config()

        # mock the model run
        model = self._get_test_model(config.architecture, config)
        model.run()

        # get the outputdata
        unique_layers = self._get_unique_layers(config)
        image_paths = (
            [config.NO_IMG_PROMPT]
            if len(config.image_paths) == 0 else
            config.image_paths
        )
        tensors = []
        for layer in unique_layers:
            for image_path in image_paths:
                query_img_path = (
                    os.path.abspath(image_path)
                    if image_path != config.NO_IMG_PROMPT else
                    image_path
                )
                tensor = self._retrieve_tensors(config, layer, query_img_path)

                if isinstance(tensor, list) and isinstance(tensor[0], tuple):
                    tensors.append(self._get_last_timestamp(tensor))
                else:
                    tensors.append(tensor)

        # TODO: rename the tensors into tuples to be more specific

        return tensors

    def _get_last_timestamp(self, tensors: List[Tuple[str, torch.Tensor, str, str, str]]) -> None:
        """Get the last timestamp from the tensors."""
        latest_timestamp = max([timestamp for _, _, timestamp, _, _ in tensors])

        # Filter the tensors with the latest timestamp
        for layer, tensor, timestamp, image_path, prompt in tensors:
            if timestamp == latest_timestamp:
                return layer, tensor, timestamp, image_path, prompt

    def test_input_ids(self, config_path: str) -> None:
        """."""
        self._import()
        pass

    def test_hidden_states(self, config_path: str) -> None:
        """Test if hidden states are floats."""
        self._import()

        # TODO: mock a model config for each test

        # get the model output
        # TODO: to save the model output globaly to save time
        model_outputs = self._model_run(config_path=config_path)

        # check that input_ids are integers
        for model_output in model_outputs:
            _, tensors, _, _, _ = model_output
            for tensor in tensors:
                # Check if the tensor is a float
                assert tensor.dtype in [torch.float, torch.float32, torch.float64, torch.bfloat16], \
                    f'Tensor is not a float: {tensor.dtype}'

    def test_hidden_states_same_input(self, config_path: str) -> None:
        """Check if the same input generates the same outputs."""
        self._import()

        pass
