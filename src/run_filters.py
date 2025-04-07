"""main.py.

This module here is the entrypoint to the VLM Competence toolkit.
"""
import argparse
import logging

import yaml

from main import get_model
from models.config import Config


class FilterConfig():
    """Config class for using different prompts during filtering."""

    def __init__(self):
        """Constructs the different filters and sets self.filters."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-fc',
            '--filter-config',
            type=str,
            required=True,
            help='The config to the filters for multiple runs'
        )
        # get the args and throw out what we don't know
        args = parser.parse_known_args()[0]

        with open(args.filter_config, 'r') as filter_file:
            self.filter_data = yaml.safe_load(filter_file)


def run_filters(base_config: Config, filter_config: FilterConfig):
    """Read through each filter config, modify config accordingly, and execute.

    This will save a different set of tensors for each filter given.

    Args:
        base_config (Config): The base config.
        filter_config (FilterConfig): The config containing all the filters.
    """
    model = get_model(base_config.architecture, base_config)

    # now for each filter, we want to modify the model's config
    for key, data in filter_config.filter_data.items():
        logging.debug(f'Running {key} filter, setting data to {data}')
        model.config.set_image_paths(data['input_dir'])
        model.config.set_modules(data['modules'])
        model.config.set_prompt(data['prompt'])
        model.run()


if __name__ == '__main__':
    # first grab the base config
    logging.getLogger().setLevel(logging.INFO)
    config = Config()
    logging.debug(
        f'Config is set to '
        f'{[(key, value) for key, value in config.__dict__.items()]}'
    )

    # then grab the filtered config
    filter_config = FilterConfig()

    # finally run the filters as needed
    run_filters(config, filter_config)
