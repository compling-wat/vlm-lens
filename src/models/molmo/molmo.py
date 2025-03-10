"""molmo.py.

File for providing the Molmo model implementation.
"""
import logging
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from .base import ModelBase
from .config import Config, ModelSelection


class MolmoModel(ModelBase):
    """Molmo model implementation."""

    def __init__(self, config: Config):
        """Initialization of the molmo model.

        This makes sure to set the model name, path and config.

        Args:
            config (Config): Parsed config
        """
        self.model_name = ModelSelection.MOLMO
        self.config = config

        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **self.config.model, trust_remote_code=True
        ) if hasattr(self.config, 'model') else (
            AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        )

    def _init_processor(self) -> None:
        """Initializes the processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
        )

    def _call_processor(self) -> BatchFeature:
        """Call processor to process the input data."""
        logging.debug('Generating embeddings ... ')

        # generate the inputs
        inputs = self.processor.process(
            text=self.config.prompt,
            images=[Image.open(os.path.join(self.config.input_dir, image)) for image in os.listdir(self.config.input_dir)],
            )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        return inputs
