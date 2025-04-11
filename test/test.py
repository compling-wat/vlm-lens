"""test/test.py.

This module tests the functionality of hidden state extraction model classes.
"""
import importlib
import os
import shutil
import sys
from typing import List, Tuple

import pytest
import torch

# NOTE: modify your number of images for bulk testing here
N_IMAGES = 2


# NOTE: when testing your model, add your config file here
@pytest.mark.parametrize(
    ('config_path'),
    [
        # 'configs/qwen-2b.yaml',
        'configs/llava-7b.yaml',
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

    # NOTE: add your model imports here
    def _get_model(self, model_arch, config):
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
        tensors_tuples = []
        for image_path in image_paths:
            query_img_path = (
                os.path.abspath(image_path)
                if image_path != config.NO_IMG_PROMPT else
                image_path
            )
            for layer in unique_layers:
                # get the tensors
                tensor_tuple = self._retrieve_tensors(config, layer, query_img_path, True)
                tensors_tuples.extend(tensor_tuple)

        if group_by == 'layer':
            tensors_tuples = sorted(tensors_tuples, key=lambda x: x[0])
        elif group_by == 'image_path':
            tensors_tuples = sorted(tensors_tuples, key=lambda x: x[3])

        return tensors_tuples

    def test_hidden_states(self, config_path: str) -> None:
        """Test if hidden states are floats."""
        self._import()

        # init config
        image_path_original = 'data'
        image_path = 'test/single'
        image_name = 'black_in_black.png'
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        if not os.path.exists(os.path.join(image_path, image_name)):
            os.symlink(
                os.path.abspath(os.path.join(image_path_original, image_name)),
                os.path.join(image_path, image_name)
                )

        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config = self.Config()

        # get the model output
        model_outputs = self._model_run(config=config)

        # check if the tensor is a float
        for model_output in model_outputs:
            _, tensors, _, _, _ = model_output
            for tensor in tensors:
                assert tensor.dtype in [torch.float, torch.float32, torch.float64, torch.bfloat16], \
                    f'Tensor is not a float: {tensor.dtype}'

        # remove the test folder
        shutil.rmtree(image_path)

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
            if not os.path.exists(os.path.join(image_path, f'copied_{i}.png')):
                os.symlink(
                    os.path.abspath(image_filename),
                    os.path.join(image_path, f'copied_{i}.png')
                    )

    def test_hidden_states_same_input(self, config_path: str) -> None:
        """Check if the same input generates the same outputs."""
        self._import()

        # create test images
        image_path_original = 'data'
        image_path = 'test/single'
        image_name = 'black_in_black.png'
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        if not os.path.exists(os.path.join(image_path, image_name)):
            os.symlink(
                os.path.abspath(os.path.join(image_path_original, image_name)),
                os.path.join(image_path, image_name)
                )

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

        # remove the test folder
        shutil.rmtree(image_path)

    def _grouper(self, n: int, iterable: list) -> iter:
        """Collect data into fixed-length chunks or blocks of adjacent elements.

        Args:
            n (int): Number of elements per chunk (here it would be 2)
            iterable (list): The iterable to process

        Returns:
            An iterator yielding tuples of adjacent elements from the iterable
        """
        # convert iterable to iterator to handle both lists and iterators
        iterator = iter(iterable)

        # yield pairs of elements
        while True:
            # get two consecutive elements
            batch = []
            for _ in range(n):
                try:
                    batch.append(next(iterator))
                except StopIteration:
                    if batch:  # if we have at least one element
                        yield batch
                    return

            if batch:  # if we have collected elements
                yield batch

    def test_hidden_states_diff_inputs(self, config_path: str) -> None:
        """Check if the different inputs generate different outputs."""
        self._import()

        # create test images
        image_path_original = 'data'
        image_path = 'test/diff'
        image_name_1 = 'black_in_black.png'
        image_name_2 = 'black_in_blue.png'
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        if not os.path.exists(os.path.join(image_path, image_name_1)):
            os.symlink(
                os.path.abspath(os.path.join(image_path_original, image_name_1)),
                os.path.join(image_path, image_name_1)
                )
        if not os.path.exists(os.path.join(image_path, image_name_2)):
            os.symlink(
                os.path.abspath(os.path.join(image_path_original, image_name_2)),
                os.path.join(image_path, image_name_2)
                )

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

        # remove the test folder
        shutil.rmtree(image_path)

    def test_in_bulk(self, config_path: str) -> None:
        """Test if the toolkit works with a bulk of images."""
        self._import()

        # create test images
        image_path_original = 'data'
        image_path = 'test/bulk'
        image_name = 'black_in_black.png'
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        if not os.path.exists(os.path.join(image_path, image_name)):
            os.symlink(
                os.path.abspath(os.path.join(image_path_original, image_name)),
                os.path.join(image_path, image_name)
                )

        self._copy_image(image_path, N_IMAGES - 1)

        # get model outputs
        sys.argv = ['test/test.py',
                    '-c', config_path,
                    '-i', image_path,
                    '--debug'
                    ]
        config = self.Config()
        model_outputs = self._model_run(config=config)

        # check if the tensors are floats
        for model_output in model_outputs:
            _, tensors, _, _, _ = model_output
            for tensor in tensors:
                assert tensor.dtype in [torch.float, torch.float32, torch.float64, torch.bfloat16], \
                    f'Tensor is not a float: {tensor.dtype}'

        # remove the test folder
        shutil.rmtree(image_path)
