"""Download images from a given query with deduplication.

This module provides functionality to download Creative Commons licensed images
using YAML configuration.
"""

import os
import sys
from typing import Optional

import yaml

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from config import DownloadConfig, ImageFormat, SearchProvider  # noqa: E402
from license import LicenseDownloader  # noqa: E402


def load_config_from_yaml(yaml_path: str) -> DownloadConfig:
    """Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        DownloadConfig instance.
    """
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Convert string enums to actual enums
    if 'search_provider' in config_data:
        provider_map = {
            'google': SearchProvider.GOOGLE,
            'bing': SearchProvider.BING,
            'both': SearchProvider.BOTH
        }
        config_data['search_provider'] = provider_map.get(
            config_data['search_provider'].lower(),
            SearchProvider.BOTH
        )

    if 'image_format' in config_data:
        format_map = {
            'jpeg': ImageFormat.JPEG,
            'png': ImageFormat.PNG,
            'webp': ImageFormat.WEBP
        }
        config_data['image_format'] = format_map.get(
            config_data['image_format'].lower(),
            ImageFormat.JPEG
        )

    # Convert list to tuple for direct_img_extensions
    if 'direct_img_extensions' in config_data:
        config_data['direct_img_extensions'] = tuple(config_data['direct_img_extensions'])

    return DownloadConfig(**config_data)


def main(config_path: Optional[str] = None) -> None:
    """Main function to download images using YAML configuration.

    Args:
        config_path: Path to YAML config file. If None, uses 'colors.yaml'.
    """
    if config_path is None:
        config_path = 'colors.yaml'

    if not os.path.exists(config_path):
        print(f'❌ Configuration file not found: {config_path}')
        return

    try:
        config = load_config_from_yaml(config_path)
        print(f'✓ Loaded configuration from: {config_path}')
    except Exception as e:
        print(f'❌ Error loading configuration: {e}')
        return

    # Create downloader and run
    downloader = LicenseDownloader(config)
    downloader.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download Creative Commons images')
    parser.add_argument(
        '--config',
        type=str,
        default='colors.yaml',
        help='Path to YAML configuration file (default: colors.yaml)'
    )

    args = parser.parse_args()
    main(args.config)
