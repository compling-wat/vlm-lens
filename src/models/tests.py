"""Unit tests for the model classes."""
import torch

# NOTE: import your model here
from main import get_model
from models.config import Config


class TestModel:
    """Tests for testing the validation of the model classes."""
    def __init__(self,
                 config: Config,
                 check_input_ids: bool = True,
                 check_mixed_input: bool = True,
                 input_path: str = 'data/',
                 output_path: str = 'output_dir/'
                 ):
        """Main thread to run the tests."""
        # configs
        self.config = config
        self.model = get_model(config.architecture, config)

        # inputs
        # TODO: remove hard-coding by adding a test-config file
        self.input_path = input_path
        self.output_path = output_path

        # flags
        self._check_input_ids = check_input_ids
        self._check_mixed_input = check_mixed_input

        # run all tests
        self.run_all()

    def _run_all(self):
        """Run all tests."""
        # tests on input ids
        if self._check_input_ids:
            self.check_input_ids()

        # tests on hidden states
        self.check_hidden_states()
        self.check_same_input()
        self.check_different_input()
        self.check_input_different_size()
        self.check_input_batch()

        # tests mixed input (text-only and image-only)
        if self._check_mixed_input:
            self.check_image_only()
            self.check_text_only()

    def _load_input_ids(self, path_input_ids: str) -> None:
        """Load the input ids from a given path."""
        input_ids = torch.load(path_input_ids)
        return input_ids

    def _load_embeddings(self, path_embeddings: str) -> None:
        """Load the embeddings from a given path."""
        # TODO: check the dimensions of the output embeddings for different models
        embeddings = torch.load(path_embeddings)
        return embeddings

    def _load_embeddings_db(self, path_embeddings: str) -> None:
        """Load the embeddings from a given database path."""
        # TODO: check the dimensions of the output embeddings for different models
        pass

    def check_input_ids(self):
        """Check if input ids are integers."""
        inputs = self.model.load_input_data()
        assert inputs['input_ids'].dtype == torch.int64

    def check_hidden_states(self):
        """Check if hidden states are tensors of floats (by torch loading)."""
        self.model.run()
        output = self._load_embeddings()

        assert output.dtype == torch.float32

    def check_same_input(self):
        """Check if the same input data produces the same output."""
        # load input data and get output
        input_data1 = self.model.load_input_data()
        self.model.run(input_data1)
        output1 = self._load_embeddings()
        input_data2 = self.model.load_input_data()
        output2 = self._load_embeddings()

        # assertions
        assert torch.equal(input_data1, input_data2)
        assert torch.equal(output1, output2)

    def check_different_input(self):
        """Check if different input data produces different output."""
        pass

    def check_input_different_size(self):
        """Check if input images of different sizes get the same output shape."""
        pass

    def check_input_batch(self):
        """Check if input ids of different batch sizes get the same output shape."""
        pass

    def check_image_only(self):
        """Check if the model works on image only inputs."""
        pass

    def check_text_only(self):
        """Check if the model works on text only inputs."""
        pass

    # TODO: stress test


if __name__ == '__main__':
    config = Config()
    TestModel(config, check_input_ids=True, check_mixed_input=True)
