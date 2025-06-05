"""Utility functions for loading dataset."""

import os
import pathlib
import sys
from typing import Optional

import datasets
import wget
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]  # go up two levels: dataset -> src -> root
sys.path.append(str(ROOT))
from project_root import PROJECT_ROOT  # noqa: E402


def process_path(path: str, to_str: bool = False) -> pathlib.Path | str:
    """Process the relative path to its project root.

    Args:
        path (str): The relative path.
        to_str (bool): Whether to convert it to str type.
    """
    target = PROJECT_ROOT / path
    if to_str:
        return str(target)
    else:
        return target


def get_text_dataset(hf_path: str, split: str, save_path: Optional[str] = None) -> datasets.Dataset:
    """Load a split of a certain dataset from HF. If save_path is specified, save the dataset locally as well.

    Args:
        hf_path (str): The huggingface path of the dataset.
        split (str): The split name.
        save_path (str | None): the disk location to save the dataset. If is None, do not save it locally.
    """
    dataset = datasets.load_dataset(hf_path)[split]
    if save_path is not None:
        dataset.save_to_disk(process_path(save_path))
    return dataset


def download_file(url: str, dst_path: str) -> None:
    """Download image zip file.

    Args:
        url (str): The url to download from.
        dst_path (str): The location to save the filw. Should be a filename instead of dir name.
    """
    dst_path = process_path(dst_path, to_str=True)
    print(f'===Downloading from {url} to {dst_path}===')
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    wget.download(url, out=dst_path)


def main(yaml_path: str):
    """Main function.

    Args:
        yaml_path (str): The path of yaml config file
    """
    yaml_path = process_path(yaml_path)
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    os.makedirs(process_path(config['parent_folder']), exist_ok=True)
    print('===step 1/2: Downloading text dataset===')
    get_text_dataset(config['dataset_path'], config['split_name'], config['dataset_download_place'])
    print('===step 2/2: Downloading image dataset===')
    download_file(config['img_url'], config['img_download_place'])


if __name__ == '__main__':
    name = sys.argv[1]
    main(name)
