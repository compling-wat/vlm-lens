"""Utility functions to map text datasets to images."""

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

    Returns:
        datasets.Dataset: A new dataset with text and images mapped together.
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

    # Map the text dataset entries to their corresponding images
    mapped_dataset = mapped_dataset.map(lambda x: {
        'image_path': filename_to_path.get(image_regex.format(id=str(x['id']).replace(',', '')), None)
    }).filter(lambda x: x['image_path'] is not None)

    # Save dataset if path provided
    if save_path:
        mapped_dataset.save_to_disk(save_path)

    return mapped_dataset


def main(config_path: str):
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
    # Map text datasets to image paths
    map_text_to_images(
        text_dataset,
        image_paths,
        image_column=config['image_column'],
        prompt_column=config['prompt_column'],
        label_column=config.get('label_column', None),
        image_regex=config.get('image_regex', '{id}'),
        save_path=config.get('save_path', None),
    )


if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)
