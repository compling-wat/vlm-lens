"""Utility functions for loading dataset."""

import os
import sys
from typing import Optional

import datasets
import wget
import yaml


def get_text_dataset(hf_path: str, split: str, save_path: Optional[str] = None) -> datasets.Dataset:
    """Load a split of a certain dataset from HF. If save_path is specified, save the dataset locally as well."""
    dataset = datasets.load_dataset(hf_path)[split]
    if save_path is not None:
        dataset.save_to_disk(save_path)
    return dataset


def download_file(url, dst_path):
    """Download image zip file."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    wget.download(url, out=dst_path)


def main(yaml_path):
    """Main function."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    os.makedirs(config['parent_folder'], exist_ok=True)
    get_text_dataset(config['dataset_path'], config['split_name'], config['dataset_download_place'])
    download_file(config['img_url'], config['img_download_place'])


if __name__ == '__main__':
    name = sys.argv[1]
    main(name)
