"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum

from transformers import AutoProcessor


class ModelSelection(str, Enum):
    """Enum that contains all possible model choices."""
    LLAVA = 'llava'
    QWEN = 'qwen'


class ModelBase(ABC):
    """Provides an abstract base class for everything to implement."""

    def __init__(self):
        """Initialization of the model base class."""
        assert self.model_path is not None
        self.load_model()

    def load_model(self):
        """Loads the model and sets the processor from the loaded model."""
        logging.debug(
            f'Loading model {self.model_name.value}; {self.model_path}'
        )
        self.load_specific_model()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    @abstractmethod
    def load_specific_model(self):
        """Abstract method to be implemented by each subclass."""
        pass

    @abstractmethod
    def classify_input_ids(self):
        """Abstract method to be implemented by each subclass."""
        pass
