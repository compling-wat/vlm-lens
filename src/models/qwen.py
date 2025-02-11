"""qwen.py.

File for providing the Qwen model implementation.
"""
from transformers import Qwen2VLForConditionalGeneration

from .base import ModelBase
from .config import Config


class QwenModel(ModelBase):
    """Qwen model implementation."""

    def __init__(self, config: Config):
        """Initialization of the qwen model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self):
        """Overridden function to populate self.model."""
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, **self.config.model
        ) if hasattr(self.config, 'model') else (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path
            )
        )

    def classify_input_ids(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Separate image and text inputs and convert them to input ids.

        Args:
            inputs          : image and text inputs.

        Returns:
            input_ids       : image and text input ids.
        """
        # load models
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # decode input ids
        input_ids = {}
        if inputs['img']:
            image_inputs = self.image_processor(inputs['img'], return_tensors='pt')
            input_ids['img'] = image_inputs['pixel_values']
        if inputs['txt']:
            text_inputs = self.tokenizer(inputs['txt'], return_tensors='pt')
            input_ids['txt'] = text_inputs['input_ids']

        return input_ids
