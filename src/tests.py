"""Unit tests for the model classes."""
import os

import torch

# NOTE: import your model here
from main import get_model
from models.config import Config


class TestModel:
    """Tests for testing the validation of the model classes."""
    def __init__(self, config, check_input_ids=True, check_mixed_input=True):
        """Main thread to run the tests."""
        self.config = config
        self.model = get_model(config.architecture, config)

        if check_input_ids:
            self.check_input_ids()

        self.check_hidden_states()
        self.check_same_input()
        self.check_different_input()
        self.check_input_different_size()
        self.check_input_batch()

        if check_mixed_input:
            self.check_image_only()
            self.check_text_only()

    def check_input_ids(self):
        """Check if input ids are integers."""
        inputs = self.model.load_input_data()
        assert inputs['input_ids'].dtype == torch.int64

    def check_hidden_states(self):
        """Check if hidden states are tensors of floats (by torch loading)."""
        self.model.run()
        for filename in os.listdir(self.config.output_dir):
            hidden_states = torch.load(os.path.join(self.config.output_dir, filename))
            assert hidden_states.dtype == torch.float32

    def check_same_input(self):
        """Check if the same input data produces the same output."""
        # load input data and get output
        input_data1 = self.model.load_input_data()
        output1 = self.model.forward(input_data1)
        input_data2 = self.model.load_input_data()
        output2 = self.model.forward(input_data2)

        # assertions
        assert torch.equal(input_data1, input_data2)
        assert torch.equal(output1, output2)

    def check_different_input(self):
        """Check if different input data produces different output.

        TODO: how to differentiate different inputs?
        """
        pass

    def check_input_different_size(self):
        """Check if input images of different sizes get the same output shape.

        TODO: how to differentiate different inputs?
        """
        pass

    def check_input_batch(self):
        """Check if input ids of different batch sizes get the same output shape.

        TODO: how to differentiate different inputs?
        """
        pass

    def check_image_only(self):
        """Check if the model works on image only inputs."""
        pass

    def check_text_only(self):
        """Check if the model works on text only inputs."""
        pass


if __name__ == '__main__':
    config = Config()
    TestModel(config, check_input_ids=True, check_mixed_input=True)
