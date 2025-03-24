"""clip.py.

File for providing the Clip model implementation.
"""

from transformers import CLIPModel

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

    def _generate_prompt(
        self,
        img_filter: dict
    ) -> str:
        """Generates the CLIP model prompt which will not use the chat template.

        Args:
            img_filter (dict): The image filter that specifies the image and
                text inputs.

        Returns:
            torch.Tensor: The data as a torch tensor.
        """
        return img_filter['prompt'] if 'prompt' in img_filter else None
