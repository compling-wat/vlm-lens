"""Utility functions to map text datasets to images."""

import importlib.util
import os
import pathlib
import sys
from typing import List, Optional

import yaml
from datasets import Dataset, load_dataset

# go up two levels: dataset -> scripts -> root
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from project_root import PROJECT_ROOT  # noqa: E402


def map_text_to_images(
    text_dataset: Dataset,
    image_paths: List[str],
    image_column: str,
    prompt_column: str,
    label_column: Optional[str] = None,
    image_regex: Optional[str] = '{id}',
    save_path: Optional[str] = None,
    map_fn: Optional[callable] = None,
) -> Dataset:
    """Map text dataset to image dataset.

    Args:
        text_dataset (Dataset): The text dataset.
        image_paths (List[str]): List of the input image file paths.
        image_column (str): The column name in text_dataset used to match entries in image_dataset.
        prompt_column (str): The column name for the prompt entry in text_dataset.
        label_column (Optional[str]): The column name for the classification label/answer entry in text_dataset.
        image_regex (Optional[str]): The regex pattern to convert image_column -> image filename.
        save_path (Optional[str]): The location to save the dataset.
        map_fn (Optional[callable]): A custom mapping function.

    Returns:
        Dataset: A new dataset with text and images mapped together.
    """
    # Create a lookup of filename -> full path
    filename_to_path = {
        path.split('/')[-1]: path
        for path in image_paths
    }

    # Create a new dataset mapping text entries to their corresponding images
    mapped_dataset = Dataset.from_dict({
        'id': text_dataset[image_column],
        'prompt': text_dataset[prompt_column]
    })

    # If the ground truth answer is provided, add it to dataset
    if label_column:
        mapped_dataset = mapped_dataset.add_column(
            'label', text_dataset[label_column]
        )

    def resolve_filename(row: dict) -> dict:
        """Helper function to define the mapping method for the image id to its path.

        Args:
            row (dict): The input row from the text dataset.

        Returns:
            dict: A dictionary containing the resolved image path.

        Raises:
            ValueError: If the mapping function encounters an error.
        """
        if map_fn:
            # Use custom map function is provided
            try:
                match_key = map_fn(row['id'])
            except Exception as e:
                raise ValueError(
                    f'Error encountered in custom mapping function: {e}')
        else:
            # Define mapping function using input regex
            clean_id = str(row['id']).replace(',', '')
            match_key = image_regex.format(id=clean_id)

        return {'image_path': filename_to_path.get(match_key, None)}

    # Map the text dataset entries to their corresponding images
    mapped_dataset = mapped_dataset.map(resolve_filename)
    mapped_dataset = mapped_dataset.filter(
        lambda x: x['image_path'] is not None)

    # Save dataset if path provided
    if save_path:
        mapped_dataset.save_to_disk(save_path)

    return mapped_dataset


def load_function_from_file(file_path: str, fn_name: str) -> callable:
    """Loads a function from an input file_path.

    Args:
        file_path (str): The path to the file containing the function.
        fn_name (str): The name of the function to load.

    Returns:
        callable: The loaded function.
    """
    spec = importlib.util.spec_from_file_location('map_module', file_path)
    map_module = importlib.util.module_from_spec(spec)
    sys.modules['map_module'] = map_module
    spec.loader.exec_module(map_module)
    return getattr(map_module, fn_name)


def main(config_path: str) -> None:
    """Main function to run the dataset mapper.

    Args:
        config_path (str): The yaml file containing the config attributes for the script/
    """
    yaml_path = os.path.join(PROJECT_ROOT, config_path)
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load text_dataset
    text_dataset = load_dataset(config['text_dataset_path'])[
        config['text_split']]

    # Locate image dataset dir
    image_dir = config.get('image_dataset_path', None)
    if image_dir and config.get('image_split', None):
        image_dir = os.path.join(image_dir, config['image_split'])

    image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_paths = [
        os.path.join(image_dir, img_path)
        for img_path in filter(
            lambda file_path:
                os.path.splitext(file_path)[1].lower() in image_exts,
            os.listdir(image_dir)
        )
    ]
    custom_map_fn = None
    if 'custom_mapping' in config:
        file_path = config['custom_mapping'].get('file', None)
        fn_name = config['custom_mapping'].get('function', None)

        assert file_path is not None and fn_name is not None, (
            'Must declare both the function name and its path under `file` and `function`.'
        )

        custom_map_fn = load_function_from_file(file_path, fn_name)

        # Map text datasets to image paths
    map_text_to_images(
        text_dataset,
        image_paths,
        image_column=config['image_column'],
        prompt_column=config['prompt_column'],
        label_column=config.get('label_column', None),
        image_regex=config.get('image_regex', '{id}'),
        save_path=config.get('save_path', None),
        map_fn=custom_map_fn,
    )


if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)
