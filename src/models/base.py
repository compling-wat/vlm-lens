"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""
import io
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Tuple, TypeAlias

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from .config import Config

ModelInput: TypeAlias = Tuple[str, str, BatchFeature]


class ModelBase(ABC):
    """Provides an abstract base class for everything to implement."""

    def __init__(self, config: Config):
        """Initialization of the model base class.

        Args:
            config (Config): Parsed config.
        """
        self.model_path = config.model_path
        self.config = config

        # log the modules -- note that this causes an exit
        if self.config.log_named_modules:
            self._log_named_modules()
            exit(0)

        # load the specific model
        logging.debug(
            f'Loading model {self.config.architecture.value}; {self.model_path}'
        )
        self._load_specific_model()

        # load the processor
        self._init_processor()

    def _log_named_modules(self):
        """Logs the named modules based on the loaded model."""
        file_path = 'logs/' + self.model_path + '.txt'
        directory_path = os.path.dirname(file_path)

        # if the path exists to the file, don't load the model again
        if os.path.isfile(file_path):
            logging.debug(f'Named modules are cached in {file_path}')
            return

        # in which case, we first load the model, then output its modules
        self._load_specific_model()

        # otherwise, we log the output to that file, and creating directories
        # as needed
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(file_path, 'w') as output_file:
            output_file.writelines(
                [f'{name}\n' for name, _ in self.model.named_modules()]
            )

    @abstractmethod
    def _load_specific_model(self):
        """Abstract method that loads the specific model."""
        pass

    def _init_processor(self) -> None:
        """Initialize the self.processor by loading from the path."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _generate_state_hook(self, name: str, image_path: str, prompt: str):
        """Generates the state hook depending on the embedding type.

        Args:
            name (str): The module name.
            image_path (str): The path to the image used for the specific pass.
            prompt (str): The prompt used for the specific pass.

        Returns:
            hook function: The hook function to return.
        """
        # first, let's modify image path to be an absolute path
        if image_path != self.config.NO_IMG_PROMPT:
            image_path = os.path.abspath(image_path)

            # this image path should already exist, error out if someone isn't
            # properly providing an image path
            assert os.path.exists(image_path)

        def generate_states_hook(module, input, output):
            """Hook handle function that saves the embedding output to a tensor.

            This tensor will be saved within a SQL database, according to the
            connection that was initialized previously.

            Args:
                module: The module that save its hook on.
                input: The input used.
                output: The embeddings to save.
            """
            cursor = self.connection.cursor()

            # Convert the tensor to a binary blob
            tensor_blob = io.BytesIO()
            torch.save(output, tensor_blob)

            # Insert the tensor into the table
            cursor.execute(f"""
                    INSERT INTO {self.config.DB_TABLE_NAME}
                    (name, architecture, image_path, prompt, layer, tensor)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (
                    self.model_path,
                    self.config.architecture.value,
                    image_path,
                    prompt,
                    name,
                    tensor_blob.getvalue()
                )
            )

            self.connection.commit()

            logging.debug(
                f'Ran hook and saved tensor for {image_path} using prompt '
                f'{prompt} on layer {name}.'
            )

        return generate_states_hook

    def _register_module_hooks(
        self,
        image_path: str,
        prompt: str
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register the generated hook function to the modules in the config.

        At the same time, we need to add in the image path itself and the prompt
        which will be used for the database input.

        Args:
            image_path (str): The path to the image used for the specific pass.
            prompt (str): The prompt used for the specific pass.

        Raises:
            RuntimeError: Calls a runtime error if no hooks were registered

        Returns:
            List[torch.utils.hooks.RemovableHandle]: A list of handles that one
                can remove after the forward pass.
        """
        logging.debug(
            f'Registering module hook for {image_path} using prompt "{prompt}"'
        )

        # a list of hooks to remove after the forward pass
        hooks = []

        # for each module, register the state hook, which will save the output
        # state from the module to a sql database, according to above
        for name, module in self.model.named_modules():
            if self.config.matches_module(name):
                hooks.append(module.register_forward_hook(
                    self._generate_state_hook(name, image_path, prompt)
                ))
                logging.debug(f'Registered hook to {name}')

        if len(hooks) == 0:
            raise RuntimeError(
                'No hooks were registered. Double-check the configured modules.'
            )

        return hooks

    def _forward(self, data: BatchFeature):
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (BatchFeature): The given data tensor.
        """
        data.to(self.config.device)
        with torch.no_grad():
            _ = self.model(**data)
        logging.debug('Completed forward pass...')

    def _hook_and_eval(self, input: ModelInput):
        """Given some input, performs a single forward pass.

        Args:
            input (ModelInput): The tuple of the image path, prompt and
                input data dictionary.
        """
        logging.debug('Starting forward pass')
        self.model.eval()

        image_path, prompt, data = input

        # now set up the modules to register the hook to
        hooks = self._register_module_hooks(image_path, prompt)

        # then ensure that the data is correct
        self._forward(data)

        for hook in hooks:
            hook.remove()
        logging.debug('Unregistered all hooks..')

    def _initialize_db(self):
        """Initializes a database based on config."""
        # Connect to the database, creating it if it doesn't exist
        self.connection = sqlite3.connect(self.config.output_db)
        logging.debug(f'Database created at {self.config.output_db}')

        cursor = self.connection.cursor()

        # Create a table
        cursor.execute(
            f"""
                CREATE TABLE IF NOT EXISTS {self.config.DB_TABLE_NAME} (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    tensor BLOB NOT NULL
                );
            """
        )
        self.connection.commit()

    def _cleanup(self):
        """Cleanups the database by closing the connection."""
        self.connection.close()

    def _generate_processor_output(self, prompt, img_path) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        return self.processor(**(
            {
                'text': prompt,
                'return_tensors': 'pt'
            }
            if img_path is None else
            {
                'text': prompt,
                'images': [Image.open(img_path).convert('RGB')],
                'return_tensors': 'pt'
            }
        ))

    def _generate_prompt(self, add_generation_prompt: bool = True) -> str:
        """Generates the prompt string with the input messages.

        Args:
            add_generation_prompt (bool): Whether to add a start token of a bot
                response.
            TODO: move `add_generation_prompt` to the config.

        Returns:
            str: The generated prompt with the input text and the image labels.
        """
        logging.debug('Loading data...')

        # build the input dict for the chat template
        input_msgs_formatted = [{
            'role': 'user',
            'content': []
        }]

        # add the image if it exists
        if self.config.has_images():
            input_msgs_formatted[0]['content'].append({
                'type': 'image'
            })

        # add the prompt if it exists
        if hasattr(self.config, 'prompt'):
            input_msgs_formatted[0]['content'].append({
                'type': 'text',
                'text': self.config.prompt
            })

        # apply the chat template to get the prompt
        return self.processor.apply_chat_template(
            input_msgs_formatted,
            add_generation_prompt=add_generation_prompt
        )

    def _load_input_data(self) -> List[ModelInput]:
        """From a configuration, loads the input image and text data.

        For each prompt and input image, create a separate batch feature that
        will be ran separately and saved separately within the database.

        Returns:
            List[ModelInput]: List of input data, this input data is made of
                a tuple of strings (first an image path, then a prompt) and
                a batch feature which is either a torch.Tensor or a dictionary.
        """
        # by default use the processor, which may not exist for each model
        logging.debug('Generating embeddings through its processor...')
        if not self.config.has_images():
            return [(
                self.config.NO_IMG_PROMPT,
                self.config.prompt,
                self._generate_processor_output(
                    prompt=self._generate_prompt(),
                    img_path=None
                )
            )]

        return [
            (
                img_path,
                self.config.prompt,
                self._generate_processor_output(
                    prompt=self._generate_prompt(),
                    img_path=img_path
                )
            )
            for img_path in self.config.image_paths
        ]

    def run(self) -> None:
        """Get the hidden states from the model and saving them."""
        # let's first initialize a database connection
        self._initialize_db()

        # then convert to gpu
        self.model.to(self.config.device)

        # then run everything else
        for input in self._load_input_data():
            self._hook_and_eval(input)

        # finally clean up, closing database connection, etc.
        self._cleanup()
