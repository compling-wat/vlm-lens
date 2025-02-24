"""clip.py.

File for providing the Clip model implementation.
"""
import os

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import ModelBase
from .config import Config


class ClipModel(ModelBase):
    """Llava model implementation."""

    def __init__(self, config: Config):
        """Initialization of the llava model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = CLIPModel.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            CLIPModel.from_pretrained(
                self.model_path
            )
        )

    def _load_processor(self):
        """Overriden function to populate self.processor."""
        self.processor = CLIPProcessor.from_pretrained(self.model_path)

    def load_input_data(self, config):
        """From a configuration, loads the input image and text data.

        Args:
            config (Config): The configuration given with image input data
            information.
            model (ModelBase): The model to use for generating the processor.

        Returns:
            torch.Tensor: The data as a torch tensor.
        """
        return self.processor(
            images=[
                Image.open(os.path.join(config.input_dir, img)).convert('RGB')
                for img in os.listdir(config.input_dir)
            ],
            text=[config.prompt for _ in os.listdir(config.input_dir)],
            return_tensors='pt'
        )
