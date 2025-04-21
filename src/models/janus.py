"""janus.py.

File for providing the Janus model implementation.
"""
from typing import List

from janus.janusflow.models import MultiModalityCausalLM as FlowCausalLM
from janus.janusflow.models import VLChatProcessor as FlowProcessor
from janus.models import MultiModalityCausalLM as BaseCausalLM
from janus.models import VLChatProcessor as BaseProcessor
from janus.utils.io import load_pil_images

from .base import ModelBase, ModelInput
from .config import Config


class JanusModel(ModelBase):
    """Janus model implementation."""

    def __init__(self, config: Config):
        """Initialize the Janus model.

        Args:
            config (Config): Parsed config.
        """
        super().__init__(config)

    def _load_specific_model(self):
        """Populate self.model with the specified Janus model."""
        self.is_pro = 'pro' in self.model_path.lower()
        self.is_flow = 'flow' in self.model_path.lower()

        model_cls = FlowCausalLM if self.is_flow else BaseCausalLM
        self.model = model_cls.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **self.config.model
        ) if hasattr(self.config, 'model') else (
            model_cls.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        )

        self.model.to(self.config.device)

    def _init_processor(self) -> None:
        """Initialize the Janus processor."""
        processor_cls = FlowProcessor if self.is_flow else BaseProcessor
        self.processor = processor_cls.from_pretrained(self.model_path)

    def _forward(self, input):
        """Perform a forward pass with the given input."""
        inputs_embeds = self.model.prepare_inputs_embeds(**input)
        return self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=input.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

    def _generate_processor_args(self):
        """Override the base function to produce processor arguments for Janus."""
        img_paths = self.config.image_paths

        user_tag = '<|User|>' if self.is_pro else 'User'
        assistant_tag = '<|Assistant|>' if self.is_pro else 'Assistant'

        conversations = []
        for img_path in img_paths:
            conversations.append([
                {
                    'role': user_tag,
                    'content': f'<image_placeholder>\n{self.config.prompt}',
                    'images': [img_path]
                },
                {
                    'role': assistant_tag,
                    'content': ''
                },
            ])

        imgs = []

        for convo in conversations:
            imgs.append(load_pil_images(convo))

        assert len(imgs) > 0, 'No images were loaded. Check your image paths and conversation structure.'

        return [
            (
                img_path,
                self.config.prompt,
                {
                    'images': img,
                    'conversations': convo,
                    'return_tensors': 'pt',
                    'padding': True,
                }
            )
            for img_path, img, convo in zip(img_paths, imgs, conversations)
        ]

    def _load_input_data(self) -> List[ModelInput]:
        """Load and preprocess batch input data from images and prompts."""
        return [
            (image_path, prompt, self.processor(**args).to(self.config.device))
            for image_path, prompt, args in self._generate_processor_args()
        ]
