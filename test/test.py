"""test/test.py.

This module tests the functionality of hidden state extraction model classes.
"""
import importlib
import os
import sys
from itertools import zip_longest
from typing import List, Tuple

import pytest
import torch
from PIL import Image


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
        """An initialization function of the test cases."""
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self._get_unique_layers = importlib.import_module('scripts.read_tensor').get_unique_layers
        self._retrieve_tensors = importlib.import_module('scripts.read_tensor').retrieve_tensors
        self.ModelBase = importlib.import_module('src.models.base').ModelBase
        self.Config = importlib.import_module('src.models.config').Config
        self.ModelSelection = importlib.import_module('src.models.config').ModelSelection
        # TODO: see if the imports are here
        self._get_model = importlib.import_moduile('src.main').get_model

    def _ensure_model(self, model):
        """Ensure the model is an object of the desired model class."""
        pass

    def _model_run(self,
                   config,
                   group_by: str = 'layer'
                   ) -> List[Tuple[str, Tuple[torch.Tensor], str, str, str]]:
        """Initialize the model and run it."""
        # mock the model run
        model = self._get_model(config.architecture, config)
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
        # TODO: move this piece of code to the read_tensor.py
        if group_by == 'layer':
            tensors = sorted(tensors, key=lambda x: x[0])
        elif group_by == 'image_path':
            tensors = sorted(tensors, key=lambda x: x[3])

        return tensors

    # TODO: move this function to the read_tensor.py
    def _get_last_timestamp(self, tensors: List[Tuple[str, torch.Tensor, str, str, str]]) -> None:
        """Get the last timestamp from the tensors."""
        latest_timestamp = max([timestamp for _, _, timestamp, _, _ in tensors])

        # Filter the tensors with the latest timestamp
        for layer, tensor, timestamp, image_path, prompt in tensors:
            if timestamp == latest_timestamp:
                return layer, tensor, timestamp, image_path, prompt

    def test_hidden_states(self, config_path: str) -> None:
        """Test if hidden states are floats."""
        self._import()

        # init config
        image_path = 'test/data/single'
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config = self.Config()

        # get the model output
        # TODO: to save the model output globaly to save time
        model_outputs = self._model_run(config=config)

        # check if the tensor is a float
        for model_output in model_outputs:
            _, tensors, _, _, _ = model_output
            for tensor in tensors:
                assert tensor.dtype in [torch.float, torch.float32, torch.float64, torch.bfloat16], \
                    f'Tensor is not a float: {tensor.dtype}'

    def _copy_image(self, image_path: str, times: int) -> None:
        """Copy the image to a different directory."""
        # check if the image path exists
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        # check if the image is already copied
        if os.path.exists(os.path.join(image_path, 'copied.png')):
            return

        # copy the image
        for i in range(times):
            image_filename = os.path.join(image_path, [filename for filename in os.listdir(image_path) if filename.endswith('.png')][0])
            image = Image.open(image_filename)
            image.save(os.path.join(image_path, f'copied_{i}.png'))

    def test_hidden_states_same_input(self, config_path: str) -> None:
        """Check if the same input generates the same outputs."""
        self._import()

        image_path = 'test/data/single'

        # get model outputs 1
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config_1 = self.Config()
        model_outputs_1 = self._model_run(config=config_1)

        # get model outputs 2
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config_2 = self.Config()
        model_outputs_2 = self._model_run(config=config_2)

        # check that the outputs are the same
        for model_output1, model_output2 in zip(model_outputs_1, model_outputs_2):
            _, tensor1, _, _, _ = model_output1
            _, tensor2, _, _, _ = model_output2

            assert torch.allclose(tensor1, tensor2), \
                f'Tensors are not close: {tensor1} and {tensor2}'

    def grouper(n, iterable, fillvalue=None):
        """Iterator to collect data into fixed-length chunks or blocks."""
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    def test_hidden_states_diff_inputs(self, config_path: str) -> None:
        """Check if the different inputs generate different outputs."""
        self._import()

        image_path = 'test/data/diff'

        # get model outputs
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config = self.Config()
        model_outputs = self._model_run(config=config)

        # check image layers
        n_layers = len(self._get_unique_layers(config))

        for model_output in self._grouper(n_layers, model_outputs):
            _, tensor1, _, _, _ = model_output[0]
            _, tensor2, _, _, _ = model_output[1]

            assert not torch.allclose(tensor1, tensor2), \
                f'Tensors are close: {tensor1} and {tensor2}'

    def test_hidden_states_diff_size(self, config_path: str) -> None:
        """Check if the different inputs generate the same outputs."""
        self._import()

        # get model output 1
        image_path_1 = 'test/data/single'
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path_1,
                    '--debug'
                    ]
        config_1 = self.Config()
        model_outputs_1 = self._model_run(config=config_1)

        # resize the image
        image_path_2 = 'test/data/diff_size'
        self._resize_image(image_path_2)

        # get model output 2
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path_2,
                    '--debug'
                    ]
        config_2 = self.Config()
        model_outputs_2 = self._model_run(config=config_2)

        # check that the outputs are the same
        for model_output1, model_output2 in zip(model_outputs_1, model_outputs_2):
            _, tensor1, _, _, _ = model_output1
            _, tensor2, _, _, _ = model_output2

            assert torch.allclose(tensor1, tensor2), \
                f'Tensors are not close: {tensor1} and {tensor2}'

    def _resize_image(self, image_path: str) -> None:
        """Resize the image to a different size."""
        # check if the image path exists
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        # check if the image is already resized
        if os.path.exists(os.path.join(image_path, 'resized.png')):
            return

        # resize the image
        image_filename = os.path.join(image_path, [filename for filename in os.listdir(image_path) if filename.endswith('.png')][0])
        image = Image.open(image_filename)
        height, width = image.size
        image = image.resize((height * 2, width * 2))
        image.save(os.path.join(image_path, 'resized.png'))

    def test_in_bulk(self, config_path: str) -> None:
        """Test if the toolkit works with a bulk of images."""
        self._import()

        # get model outputs
        image_path = 'test/data/bulk'
        self._copy_image(image_path, 10)
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config = self.Config()
        model_outputs = self._model_run(config=config)

        # use a virtual link for different images

        # check if the tensors are floats
        for model_output in model_outputs:
            _, tensors, _, _, _ = model_output
            for tensor in tensors:
                assert tensor.dtype in [torch.float, torch.float32, torch.float64, torch.bfloat16], \
                    f'Tensor is not a float: {tensor.dtype}'
