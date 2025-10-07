"""base.py.

Provides the common classes used such as the ModelSelection enum as well as the
abstract base class for models.
"""
import io
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Callable, List, Optional, TypedDict

import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from .config import Config


class ModelInput(TypedDict):
    """Definition for the general model input dictionary."""
    image: str | Image.Image
    prompt: str
    label: Optional[str]
    data: BatchFeature
    row_id: Optional[str]


class ModelBase(ABC):
    """Provides an abstract base class for everything to implement."""

    def __init__(self, config: Config) -> None:
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

    def _log_named_modules(self) -> None:
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
    def _load_specific_model(self) -> None:
        """Abstract method that loads the specific model."""
        pass

    def _init_processor(self) -> None:
        """Initialize the self.processor by loading from the path."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _generate_ablation_hook(self, head_list: List[int]) -> Callable[[torch.nn.Module, tuple, torch.Tensor | tuple], torch.Tensor | tuple]:
        """Generates the hook for ablation.

        Args:
            head_list (List[int]): The list of head to be ablated. If -1 is presented, select all heads.

        Returns:
            hook function: The hook function to return.

        """

        def generate_ablation_hook(module: torch.nn.Module, input: tuple, output: torch.Tensor | tuple) -> torch.Tensor | tuple:
            """Hook handle function that perform the ablation.

            The returned tensor will automatically replace the intermediate result used in the forward process.

            Args:
                module (torch.nn.Module): The module that save its hook on.
                input (tuple): The input used.
                output (torch.Tensor): The output embeddings to be modified.

            Returns:
                the updated output after ablation

            """
            if isinstance(output, torch.Tensor):
                attn_output = output.clone()
                if -1 in head_list:
                    attn_output[:, :, :, :] = 0.0
                else:
                    attn_output[:, head_list, :, :] = 0.0
                return attn_output
            else:
                attn_output, attn_weights, *rest = output
                attn_output = attn_output.clone()
                if -1 in head_list:
                    attn_output[:, :, :, :] = 0.0
                else:
                    attn_output[:, head_list, :, :] = 0.0
                return (attn_output, attn_weights, *rest)
        return generate_ablation_hook

    def _generate_state_hook(self,
                             name: str,
                             model_input: ModelInput
                             ) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        """Generates the state hook depending on the embedding type.

        Args:
            name (str): The module name.
            model_input (ModelInput): The input dictionary
                containing the image path, prompt, label (if applicable) and
                the data itself.

        Returns:
            hook function: The hook function to return.
        """
        image_path, prompt = model_input['image'], model_input['prompt']
        label = model_input.get('label', None)
        row_id = model_input.get('row_id', None)

        # Modify image path to be an absolute path if necessary
        if isinstance(image_path, str) and image_path != self.config.NO_IMG_PROMPT:
            image_path = os.path.abspath(image_path)

            # this image path should already exist, error out if someone isn't
            # properly providing an image path
            assert os.path.exists(image_path)

        def generate_states_hook(module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
            """Hook handle function that saves the embedding output to a tensor.

            This tensor will be saved within a SQL database, according to the
            connection that was initialized previously.

            Args:
                module (torch.nn.Module): The module that save its hook on.
                input (tuple): The input used.
                output (torch.Tensor): The embeddings to save.
            """
            if not isinstance(output, torch.Tensor):
                logging.warning(f'Output type of {str(type(module))} is not a tensor, skipped.')
                return

            cursor = self.connection.cursor()

            # Convert the tensor to a binary blob
            tensor_blob = io.BytesIO()

            # It currently averages the output across the sequence length dimension, i.e., mean pooling
            # WARNING: When contributing new models, ensure that dim 1 is always the sequence length dimension
            final_output = getattr(output, self.config.pooling_method)(dim=1) if hasattr(
                self.config, 'pooling_method') and hasattr(output, self.config.pooling_method) else output
            output_dim = final_output.shape[-1]
            torch.save(final_output, tensor_blob)

            # Insert the tensor into the table
            cursor.execute(f"""
                INSERT INTO {self.config.DB_TABLE_NAME}
                (name, architecture, image_path, image_id, prompt, label, layer, pooling_method, tensor_dim, tensor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                self.model_path,
                self.config.architecture.value,
                image_path if isinstance(image_path, str) else None,
                row_id,
                prompt,
                label,
                name,
                self.config.pooling_method if hasattr(self.config, 'pooling_method') else None,
                output_dim,
                tensor_blob.getvalue())
            )

            self.connection.commit()

            logging.debug(
                f'Ran hook and saved tensor for {image_path} using prompt '
                f'{prompt} on layer {name}.'
            )

        return generate_states_hook

    def _register_module_hooks(self,
                               model_input: ModelInput
                               ) -> List[torch.utils.hooks.RemovableHandle]:
        """Register the generated hook function to the modules in the config.

        At the same time, we need to add in the image path itself and the prompt
        which will be used for the database input.

        Args:
            model_input (ModelInput): The input dictionary
                containing the image path, prompt, label (if applicable) and
                the data itself.

        Raises:
            RuntimeError: Calls a runtime error if no hooks were registered

        Returns:
            List[torch.utils.hooks.RemovableHandle]: A list of handles that one
                can remove after the forward pass.
        """
        logging.debug(
            f'Registering module hook for {model_input["image"]} using prompt "{model_input["prompt"]}"'
        )

        # a list of hooks to remove after the forward pass
        hooks = []

        # for each module, register the state hook and save the output to database
        for name, module in self.model.named_modules():
            if self.config.matches_ablation_module(name):  # ablation is earlier
                hooks.append(module.register_forward_hook(
                    self._generate_ablation_hook(self.config.ablations[name])
                ))
                logging.debug(f'Registered ablation hook to {name}')

            if self.config.matches_module(name):
                hooks.append(module.register_forward_hook(
                    self._generate_state_hook(name, model_input)
                ))
                logging.debug(f'Registered hook to {name}')

        if len(hooks) == 0:
            raise RuntimeError(
                'No hooks were registered. Double-check the configured modules.'
            )

        return hooks

    def _forward(self, data: BatchFeature) -> None:
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

    def _hook_and_eval(self, model_input: ModelInput) -> None:
        """Given some input, performs a single forward pass.

        Args:
            model_input (ModelInput): The given input dictionary.
        """
        logging.debug('Starting forward pass')
        self.model.eval()

        # now set up the modules to register the hook to
        hooks = self._register_module_hooks(model_input)

        # then ensure that the data is correct
        self._forward(model_input['data'])

        for hook in hooks:
            hook.remove()
        logging.debug('Unregistered all hooks..')

    def _initialize_db(self) -> None:
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
                    image_path TEXT NULL,
                    image_id INTEGER NULL,
                    prompt TEXT NOT NULL,
                    label TEXT NULL,
                    layer TEXT NOT NULL,
                    pooling_method TEXT NULL,
                    tensor_dim INTEGER NOT NULL,
                    tensor BLOB NOT NULL
                );
            """
        )

    def _cleanup(self) -> None:
        """Cleanups the database by closing the connection."""
        self.connection.close()

    def _generate_processor_output(self, prompt: str, img_path: str | Image.Image) -> dict:
        """Generate the processor outputs from the prompt and image path.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str | Image.Image): The specified input image path or image object.

        Returns:
            dict: The corresponding processor output per image and prompt.
        """
        data = {
            'text': prompt,
            'return_tensors': 'pt'
        }

        if img_path:
            img = Image.open(img_path) if isinstance(img_path, str) else img_path
            data['images'] = [img.convert('RGB')]

        return self.processor(**data)

    def _generate_prompt(self, prompt: str, add_generation_prompt: bool = True, has_images: bool = False) -> str:
        """Generates the prompt string with the input messages.

        TODO: move `add_generation_prompt` to the config.
        [Note from Martin] I'd argue that we should keep it as a parameter here
        since in gradio we want to hack these parameters a bit.

        Args:
            prompt (str): The input prompt string.
            add_generation_prompt (bool): Whether to add a start token of a bot
                response.
            has_images (bool): Whether the model has images or not.

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
        if self.config.has_images() or has_images:
            input_msgs_formatted[0]['content'].append({
                'type': 'image'
            })

        # add the prompt if it exists
        if prompt:
            input_msgs_formatted[0]['content'].append({
                'type': 'text',
                'text': prompt
            })

        # apply the chat template to get the prompt
        return self.processor.apply_chat_template(
            input_msgs_formatted,
            add_generation_prompt=add_generation_prompt
        )

    def _load_input_data(self) -> Iterator[ModelInput]:
        """From a configuration, loads the input image and text data.

        For each prompt and input image, create a separate batch feature that
        will be ran separately and saved separately within the database.

        Yields:
            List[ModelInput]: List of input data, this input data is made of
                a tuple of strings (first an image path, then a prompt) and
                a batch feature which is either a torch.Tensor or a dictionary.
        """
        # by default use the processor, which may not exist for each model
        logging.debug('Generating embeddings through its processor...')
        if self.config.dataset:
            # Use the dataset to load input data, which includes (id, prompt, image_path)
            for row in self.config.dataset:
                prompt = self._generate_prompt(row['prompt'])
                data = self._generate_processor_output(
                    prompt=prompt,
                    img_path=row['image']
                )

                yield {
                    'image': row['image'],
                    'prompt': row['prompt'],
                    'label': row['label'] if 'label' in self.config.dataset.column_names else None,
                    'data': data,
                    'row_id': row['id'],
                }

        else:
            if not self.config.has_images():
                yield {
                    'image': self.config.NO_IMG_PROMPT,  # TODO: Check this?
                    'prompt': self.config.prompt,
                    'data': self._generate_processor_output(
                        prompt=self._generate_prompt(),
                        img_path=None
                    )
                }
            else:
                prompt = self._generate_prompt(self.config.prompt)
                for img_path in self.config.image_paths:
                    data = self._generate_processor_output(
                        prompt=prompt,
                        img_path=img_path
                    )
                    yield {
                        'image': img_path,
                        'prompt': self.config.prompt,
                        'data': data
                    }

    @property
    def _data_size(self) -> int:
        """Returns the total number of data points.

        Returns:
            int: The total number of data points.
        """
        if self.config.dataset:
            return len(self.config.dataset)
        else:
            if not self.config.has_images():
                return 1
            else:
                return len(self.config.image_paths)

    def run(self) -> None:
        """Get the hidden states from the model and saving them."""
        # let's first initialize a database connection
        self._initialize_db()

        # then convert to gpu
        self.model.to(self.config.device)

        # then reset the starting point in tracking maximum GPU memory, if using cuda
        if self.config.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.config.device)

        # then run everything else
        for item in tqdm.tqdm(self._load_input_data(), desc='Running forward hooks on data', total=self._data_size):
            self._hook_and_eval(item)

        # then output peak memory usage, if using cuda
        if self.config.device.type == 'cuda':
            logging.debug(f'Peak GPU memory allocated: {torch.cuda.max_memory_allocated(self.config.device) / 1e6:.2f} MB')

        # finally clean up, closing database connection, etc.
        self._cleanup()
