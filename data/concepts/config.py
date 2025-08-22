"""Configuration module for the Creative Commons Image Downloader.

This module defines the configuration class and enums for the image downloader.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SearchProvider(Enum):
    """Enum for available search providers."""
    GOOGLE = 'google'
    BING = 'bing'
    BOTH = 'both'


class ImageFormat(Enum):
    """Enum for supported image formats."""
    JPEG = 'JPEG'
    PNG = 'PNG'
    WEBP = 'WEBP'


@dataclass
class DownloadConfig:
    """Configuration class for image downloader."""

    # Basic settings
    num_images: int = 10
    normalize_size: int = 256
    data_dir: str = './data/concepts/images'
    results_filename: str = 'concepts.json'

    # Search provider configuration
    search_provider: SearchProvider = SearchProvider.BOTH
    retrieve_multiplier: int = 10

    # Image processing settings
    image_format: ImageFormat = ImageFormat.JPEG
    image_quality: int = 90
    load_truncated_images: bool = True

    # Download settings
    max_size_mb: int = 10
    timeout: int = 15

    # Deduplication settings
    enable_deduplication: bool = True
    similarity_threshold: int = 5

    # Rate limiting
    request_delay_min: float = 0.5
    request_delay_max: float = 1.5

    # Search queries
    queries: List[str] = field(default_factory=lambda: [
        'red', 'blue', 'yellow', 'green', 'orange', 'purple',
        'white', 'black', 'gray', 'silver', 'gold', 'pink',
        'brown', 'beige', 'crimson', 'maroon', 'cyan',
        'turquoise', 'violet', 'magenta'
    ])

    # User agents for rotation
    user_agents: List[str] = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ])

    # Direct image extensions
    direct_img_extensions: tuple = (
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def results_path(self) -> str:
        """Get full path to results file.

        Returns:
            str: Full path to the results file.
        """
        return os.path.join(self.data_dir, self.results_filename)

    @property
    def max_bytes(self) -> int:
        """Get maximum file size in bytes.

        Returns:
            int: Maximum file size in bytes.
        """
        return int(self.max_size_mb * 1024 * 1024)
