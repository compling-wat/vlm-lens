"""Base downloader class for image downloading functionality.

This module defines the abstract base class that all downloaders must implement.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from config import DownloadConfig


class DownloaderBase(ABC):
    """Abstract base class for image downloaders."""

    def __init__(self, config: DownloadConfig) -> None:
        """Initialize the downloader with configuration.

        Args:
            config: DownloadConfig instance containing all settings.
        """
        self.config = config
        self.downloaded_images: Dict[str, List[str]] = {}
        self.image_hashes: Set[str] = set()

    @abstractmethod
    def search_images(self, query: str) -> List[str]:
        """Search for images based on query.

        Args:
            query: Search term.

        Returns:
            List of image URLs.
        """
        pass

    @abstractmethod
    def download_image(
        self,
        url: str,
        filename: str,
        existing_hashes: Optional[Set[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Download a single image.

        Args:
            url: Image URL to download.
            filename: Local filename to save to.
            existing_hashes: Set of existing image hashes for deduplication.

        Returns:
            Tuple of (success, image_hash or None).
        """
        pass

    def run(self) -> None:
        """Main execution method - downloads images for all queries."""
        print(f'🎨 Starting image download with {len(self.config.queries)} queries...')

        for query in self.config.queries:
            print(f"\n🔍 Processing query: '{query}'")
            images = self.download_query_images(query)
            self.downloaded_images[query] = images

            if images:
                print(f'✅ Downloaded {len(images)} images for "{query}"')
            else:
                print(f'❌ No images downloaded for "{query}"')

        self.save_results()
        self.print_summary()

    def download_query_images(self, query: str) -> List[str]:
        """Download images for a single query.

        Args:
            query: Search term.

        Returns:
            List of downloaded image paths.
        """
        urls = self.search_images(query)

        if not urls:
            print(f'❌ No URLs found for query: {query}')
            return []

        print(f'📋 Found {len(urls)} potential URLs for "{query}"')

        downloaded_paths = []
        downloaded_count = 0

        for i, url in enumerate(urls):
            if downloaded_count >= self.config.num_images:
                break

            filename = self.generate_filename(query, downloaded_count + 1)
            success, img_hash = self.download_image(
                url, filename, existing_hashes=self.image_hashes
            )

            if success and img_hash:
                downloaded_paths.append(filename)
                downloaded_count += 1
                if self.config.enable_deduplication:
                    self.image_hashes.add(img_hash)

            if (i + 1) % 5 == 0:
                print(f'📊 Progress: {downloaded_count}/{self.config.num_images} downloaded')

        return downloaded_paths

    def generate_filename(self, query: str, index: int) -> str:
        """Generate filename for downloaded image.

        Args:
            query: Search query.
            index: Image index.

        Returns:
            Full path to image file.
        """
        safe_query = query.replace(' ', '_').replace('/', '_')
        extension = '.jpg' if self.config.image_format.value == 'JPEG' else '.png'
        filename = f'{safe_query}_{index:02d}{extension}'
        return os.path.join(self.config.data_dir, filename)

    def save_results(self) -> None:
        """Save download results to JSON file."""
        with open(self.config.results_path, 'w') as f:
            json.dump(self.downloaded_images, f, indent=2)

        print(f'\n💾 Results saved to: {self.config.results_path}')

    def print_summary(self) -> None:
        """Print download summary."""
        total_images = sum(len(images) for images in self.downloaded_images.values())
        print('\n🎉 Download Complete!')
        print(f'📈 Summary: {total_images} total images across {len(self.config.queries)} queries')

        for query, images in self.downloaded_images.items():
            print(f'  {query}: {len(images)} images')
