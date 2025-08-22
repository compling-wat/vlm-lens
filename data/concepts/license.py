"""Creative Commons license image downloader implementation.

This module provides the main downloader class with Google Images and Bing support.
"""

import os
import random
import re
import time
from io import BytesIO
from typing import List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, unquote

import numpy as np
import requests
from base import DownloaderBase
from bs4 import BeautifulSoup
from config import DownloadConfig, SearchProvider
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError


class LicenseDownloader(DownloaderBase):
    """Creative Commons license image downloader with Google Images and Bing support."""

    def __init__(self, config: DownloadConfig) -> None:
        """Initialize the license downloader.

        Args:
            config: DownloadConfig instance containing all settings.
        """
        super().__init__(config)
        ImageFile.LOAD_TRUNCATED_IMAGES = self.config.load_truncated_images

    def search_images(self, query: str) -> List[str]:
        """Search for Creative Commons images.

        Args:
            query: Search term.

        Returns:
            List of image URLs.
        """
        all_urls = []

        if self.config.search_provider in (SearchProvider.GOOGLE, SearchProvider.BOTH):
            google_urls = self._search_google_images(query)
            all_urls.extend(google_urls)

        if self.config.search_provider in (SearchProvider.BING, SearchProvider.BOTH):
            bing_urls = self._search_bing_images(query)
            all_urls.extend(bing_urls)

        # Deduplicate URLs
        unique_urls = list(dict.fromkeys(all_urls))

        # Filter and clean URLs
        cleaned_urls = []
        for url in unique_urls:
            cleaned = self._sanitize_google_url(url)
            if cleaned and self._is_direct_image_url(cleaned):
                cleaned_urls.append(cleaned)

        return cleaned_urls

    def _search_google_images(self, query: str) -> List[str]:
        """Search Google Images for Creative Commons licensed images.

        Args:
            query: Search term.

        Returns:
            List of candidate image URLs.
        """
        print(f"🔍 Searching Google Images for '{query}' with Creative Commons license...")
        search_query = quote(query)
        url = f'https://www.google.com/search?q={search_query}&tbm=isch&tbs=sur:fc'

        headers = {
            'User-Agent': random.choice(self.config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')

            candidates = []

            # Extract image URLs from img tags
            for img in soup.find_all('img'):
                src = img.get('data-src') or img.get('src') or ''
                if src.startswith('http') and 'gstatic' not in src:
                    candidates.append(src)

            # Extract URLs from script tags
            for script in soup.find_all('script'):
                s = script.string or ''
                for m in re.findall(r'https://[^"]+\.(?:jpg|jpeg|png|gif|webp)', s, flags=re.IGNORECASE):
                    if 'gstatic' not in m:
                        candidates.append(m)

            unique_candidates = list(dict.fromkeys(candidates))
            print(f'✓ Found {len(unique_candidates)} Google image URLs')
            return unique_candidates

        except Exception as e:
            print(f'❌ Error searching Google Images: {e}')
            return []

    def _search_bing_images(self, query: str) -> List[str]:
        """Search Bing Images for Creative Commons licensed images.

        Args:
            query: Search term.

        Returns:
            List of candidate image URLs.
        """
        print(f"🔍 Searching Bing Images for '{query}' with Creative Commons license...")
        url = f'https://www.bing.com/images/search?q={quote(query)}&qft=+filterui:license-L2_L3_L4'

        headers = {
            'User-Agent': random.choice(self.config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')

            candidates = []
            for img in soup.find_all('img', {'class': 'mimg'}):
                src = img.get('src') or ''
                if src.startswith('http'):
                    candidates.append(src)

            for img in soup.find_all('img'):
                ds = img.get('data-src') or ''
                if ds.startswith('http'):
                    candidates.append(ds)

            unique_candidates = list(dict.fromkeys(candidates))
            print(f'✓ Found {len(unique_candidates)} Bing image URLs')
            return unique_candidates

        except Exception as e:
            print(f'❌ Error searching Bing Images: {e}')
            return []

    def download_image(
        self,
        url: str,
        filename: str,
        existing_hashes: Optional[Set[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Download an image with validation and optional normalization.

        Args:
            url: URL of the image to download.
            filename: Local filename to save the image.
            existing_hashes: Optional set of image hashes for deduplication.

        Returns:
            A tuple where the first element is a boolean indicating success,
            and the second is the image hash if successful.
        """
        if existing_hashes is None:
            existing_hashes = set()

        url = self._sanitize_google_url(url)
        if not url:
            return False, None

        if re.search(r'(wikipedia|wikimedia)\.org/.*/File:', url):
            resolved = self._resolve_wikimedia_file_page(url)
            if not resolved:
                return False, None
            url = resolved

        headers = {
            'User-Agent': random.choice(self.config.user_agents),
            'Referer': 'https://www.google.com/',
            'Accept': 'image/*,*/*;q=0.8',
        }

        ok, size, ctype = self._preflight_head(url)
        if not ok:
            return False, None

        try:
            with requests.get(url, headers=headers, timeout=self.config.timeout, stream=True) as r:
                r.raise_for_status()

                buf, downloaded = BytesIO(), 0
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > self.config.max_bytes:
                        print(f'❌ Skipping {filename}: > {self.config.max_size_mb}MB')
                        return False, None
                    buf.write(chunk)

            data = buf.getvalue()
            if not data or len(data) < 64:
                return False, None

            buf.seek(0)
            try:
                with Image.open(buf) as im:
                    im.load()
                    im = ImageOps.exif_transpose(im)
                    if im.mode != 'RGB':
                        im = im.convert('RGB')

                    image_hash = self._calculate_image_hash(im)

                    if self.config.enable_deduplication:
                        for existing_hash in existing_hashes:
                            if self._images_are_similar(image_hash, existing_hash):
                                print(f'❌ Skipping {filename}: similar image exists')
                                return False, None

                    if self.config.normalize_size > 0:
                        im = self._center_crop_and_resize(im, self.config.normalize_size)

                    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
                    im.save(
                        filename,
                        format=self.config.image_format.value,
                        quality=self.config.image_quality,
                        optimize=True
                    )

                    print(f'✅ Downloaded: {filename} ({im.size[0]}x{im.size[1]})')

                    time.sleep(random.uniform(
                        self.config.request_delay_min,
                        self.config.request_delay_max
                    ))

                    return True, image_hash

            except (UnidentifiedImageError, OSError) as e:
                print(f'❌ Invalid image from {url}: {e}')
                return False, None

        except (requests.Timeout, requests.RequestException) as e:
            print(f'❌ Failed to download {url}: {e}')
            return False, None
        except Exception as e:
            print(f'❌ Unexpected error for {url}: {e}')
            return False, None

    def _is_direct_image_url(self, url: str) -> bool:
        """Check if URL appears to be a direct image link.

        Args:
            url: The URL to check.

        Returns:
            True if the URL likely points to an image, False otherwise.
        """
        u = url.split('?')[0].lower()
        if u.endswith(self.config.direct_img_extensions):
            return True
        if 'about-this-image' in url or '/imgres?' in url:
            return False
        return 'upload.wikimedia.org' in url

    def _sanitize_google_url(self, url: str) -> str:
        """Clean Google URLs and extract embedded target URLs.

        Args:
            url: Google-wrapped image URL.

        Returns:
            Cleaned image URL or the original if no cleaning was applied.
        """
        if 'about-this-image' in url:
            return ''
        if 'google' in url and 'q=' in url:
            try:
                q = parse_qs(url.split('?', 1)[1]).get('q', [''])[0]
                if q.startswith('http'):
                    return unquote(q)
            except Exception:
                pass
        return url

    def _resolve_wikimedia_file_page(self, url: str) -> str:
        """Resolve Wikimedia File: page to actual image URL.

        Args:
            url: Wikimedia File: page URL.

        Returns:
            Direct image URL if found, otherwise an empty string.
        """
        if not re.search(r'(wikipedia|wikimedia)\.org/.*/File:', url):
            return url

        try:
            r = requests.get(
                url,
                headers={'User-Agent': random.choice(self.config.user_agents)},
                timeout=10
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')

            og = soup.find('meta', {'property': 'og:image'})
            if og and og.get('content', '').startswith('http'):
                return og['content']

            cand = soup.find('a', href=re.compile(r'^https?://upload\.wikimedia\.org/'))
            if cand and cand.get('href'):
                return cand['href']

        except Exception:
            return ''
        return ''

    def _preflight_head(self, url: str) -> Tuple[bool, int, str]:
        """Perform HEAD request to validate image before download.

        Args:
            url: Image URL to validate.

        Returns:
            Tuple of (success, content length in bytes, content type).
        """
        try:
            h = requests.head(
                url,
                headers={
                    'User-Agent': random.choice(self.config.user_agents),
                    'Referer': 'https://www.google.com/',
                },
                timeout=10,
                allow_redirects=True
            )

            if h.status_code >= 400 and h.status_code not in (403, 405):
                return (False, -1, '')

            ctype = h.headers.get('Content-Type', '')
            clen = h.headers.get('Content-Length')
            size = int(clen) if (clen and clen.isdigit()) else -1

            if ctype and not ctype.lower().startswith('image/'):
                return (False, size, ctype)

            if size != -1 and size > self.config.max_bytes:
                return (False, size, ctype or '')

            return (True, size, ctype or '')

        except Exception:
            return (True, -1, '')

    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash of image for deduplication.

        Args:
            image: PIL Image object.

        Returns:
            String representing the perceptual hash.
        """
        small = image.resize((32, 32), Image.Resampling.LANCZOS)
        gray = small.convert('L')
        pixels = np.array(gray)
        avg = pixels.mean()
        binary = (pixels > avg).astype(int)
        return ''.join(str(bit) for bit in binary.flatten())

    def _images_are_similar(self, hash1: str, hash2: str) -> bool:
        """Check if two image hashes are similar.

        Args:
            hash1: First image hash.
            hash2: Second image hash.

        Returns:
            True if images are considered similar, False otherwise.
        """
        if len(hash1) != len(hash2):
            return False
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return distance <= self.config.similarity_threshold

    def _center_crop_and_resize(self, img: Image.Image, target_size: int) -> Image.Image:
        """Center crop image to square and resize.

        Args:
            img: PIL Image to crop and resize.
            target_size: Target size in pixels for the output square image.

        Returns:
            Cropped and resized PIL Image.
        """
        width, height = img.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize(
            (target_size, target_size),
            Image.Resampling.LANCZOS
        )

        return img_resized
