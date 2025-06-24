"""Utility functions to map text datasets to images."""

import os
import sys
from typing import List, Optional

import yaml
from datasets import Dataset, load_dataset

from .download_datasets import process_path


def map_text_to_images(
                        text_dataset: Dataset,
                        image_paths: List[str],
                        image_column: str,
                        prompt_column: str,
                        answer_column: Optional[str] = None,
                        save_path: Optional[str] = None
                    ) -> Dataset:
    """Map text dataset to image dataset.

    Args:
        text_dataset (datasets.Dataset): The text dataset.
        image_paths (list[str]): List of the input image file paths.
        image_column (str): The column name in text_dataset used to match entries in image_dataset.
        prompt_column (str): The column name for the prompt entry in text_dataset.
        answer_column (str): The column name for the answer entry in text_dataset.
        save_path (str): The location to save the dataset.

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
    if answer_column:
        mapped_dataset = mapped_dataset.add_column(
            'answer', text_dataset[answer_column]
        )

    # Map the text dataset entries to their corresponding images
    mapped_dataset = mapped_dataset.map(lambda x: {
        'image_path': filename_to_path.get(x['id'], None)
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
    yaml_path = process_path(config_path)
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load text_dataset
    text_dataset = load_dataset(config['text_dataset_path'])[config['text_split']]

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
        answer_column=config.get('answer_column', None),
        save_path=config.get('save_path', None),
    )


if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)
