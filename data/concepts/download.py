"""Download images from a given query with deduplication.

This module provides functionality to download Creative Commons licensed images
from Google Images and Bing, with URL and RGB-level deduplication capabilities.
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
from bs4 import BeautifulSoup
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
_DIRECT_IMG_EXT = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff')
_DATA_DIR = './images'


def create_download_folder(query: str) -> str:
    """Create a folder for downloaded images.

    Args:
        query: Search query string (used for folder organization).

    Returns:
        Path to the created folder.
    """
    folder_name = _DATA_DIR
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def get_user_agents() -> List[str]:
    """Return a list of user agents to rotate through.

    Returns:
        List of user agent strings for web requests.
    """
    return [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]


def is_direct_image_url(url: str) -> bool:
    """Check if URL appears to be a direct image link.

    Args:
        url: URL to check.

    Returns:
        True if URL has image extension or known image patterns.
    """
    u = url.split('?')[0].lower()
    if u.endswith(_DIRECT_IMG_EXT):
        return True
    # Skip known non-image endpoints
    if 'about-this-image' in url or '/imgres?' in url:
        return False
    return False


def sanitize_google_url(url: str) -> str:
    """Clean Google URLs and extract embedded target URLs.

    Args:
        url: Raw URL from Google search results.

    Returns:
        Cleaned URL or empty string if should be skipped.
    """
    if 'about-this-image' in url:
        return ''  # always skip
    # Some results may be Google redirectors carrying the real URL as a param
    if 'google' in url and 'q=' in url:
        try:
            q = parse_qs(url.split('?', 1)[1]).get('q', [''])[0]
            if q.startswith('http'):
                return unquote(q)
        except Exception:
            pass
    return url


def resolve_wikimedia_file_page(url: str, timeout: int = 10) -> str:
    """Resolve Wikimedia File: page to actual image URL.

    Args:
        url: Wikimedia File: page URL.
        timeout: Request timeout in seconds.

    Returns:
        Direct image URL or empty string on failure.
    """
    if not re.search(r'(wikipedia|wikimedia)\.org/.*/File:', url):
        return url  # not a file page
    try:
        r = requests.get(
            url,
            headers={'User-Agent': random.choice(get_user_agents())},
            timeout=timeout
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # Prefer og:image
        og = soup.find('meta', {'property': 'og:image'})
        if og and og.get('content', '').startswith('http'):
            return og['content']
        # Fallback: look for links/images pointing to upload.wikimedia.org
        cand = soup.find('a', href=re.compile(r'^https?://upload\.wikimedia\.org/'))
        if cand and cand.get('href'):
            return cand['href']
        img = soup.find('img', src=re.compile(r'^https?://upload\.wikimedia\.org/'))
        if img and img.get('src'):
            return img['src']
    except Exception:
        return ''
    return ''


def preflight_head(url: str, max_bytes: int, timeout: int = 10) -> Tuple[bool, int, str]:
    """Perform HEAD request to validate image before download.

    Args:
        url: Image URL to check.
        max_bytes: Maximum allowed file size.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (is_valid, size, content_type).
    """
    try:
        h = requests.head(
            url,
            headers={
                'User-Agent': random.choice(get_user_agents()),
                'Referer': 'https://www.google.com/',
            },
            timeout=timeout,
            allow_redirects=True
        )
        # Some servers disallow HEAD; tolerate 405/403 by falling back
        if h.status_code >= 400 and h.status_code not in (403, 405):
            return (False, -1, '')
        ctype = h.headers.get('Content-Type', '')
        clen = h.headers.get('Content-Length')
        size = int(clen) if (clen and clen.isdigit()) else -1
        if ctype and not ctype.lower().startswith('image/'):
            return (False, size, ctype)
        if size != -1 and size > max_bytes:
            return (False, size, ctype or '')
        return (True, size, ctype or '')
    except Exception:
        # Network oddities: allow GET path to try
        return (True, -1, '')


def calculate_image_hash(image: Image.Image) -> str:
    """Calculate perceptual hash of image for deduplication.

    Args:
        image: PIL Image object.

    Returns:
        Hash string for comparison.
    """
    # Resize to small size for comparison
    small = image.resize((32, 32), Image.Resampling.LANCZOS)
    # Convert to grayscale
    gray = small.convert('L')
    # Get pixel data
    pixels = np.array(gray)
    # Calculate average
    avg = pixels.mean()
    # Create hash based on whether each pixel is above/below average
    binary = (pixels > avg).astype(int)
    # Convert to string
    return ''.join(str(bit) for bit in binary.flatten())


def images_are_similar(hash1: str, hash2: str, threshold: int = 10) -> bool:
    """Check if two image hashes are similar.

    Args:
        hash1: First image hash.
        hash2: Second image hash.
        threshold: Maximum hamming distance for similarity.

    Returns:
        True if images are considered similar.
    """
    if len(hash1) != len(hash2):
        return False
    # Calculate hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    return distance <= threshold


def center_crop_and_resize(img: Image.Image, target_size: int) -> Image.Image:
    """Center crop image to square and resize.

    Args:
        img: Input PIL Image.
        target_size: Target size for both width and height.

    Returns:
        Processed PIL Image.
    """
    width, height = img.size

    # Calculate the size for center cropping (largest square that fits)
    crop_size = min(width, height)

    # Calculate crop coordinates for center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop to square
    img_cropped = img.crop((left, top, right, bottom))

    # Resize to target dimensions
    img_resized = img_cropped.resize(
        (target_size, target_size),
        Image.Resampling.LANCZOS
    )

    return img_resized


def download_image(
    url: str,
    filename: str,
    max_size_mb: int = 10,
    normalize: int = -1,
    timeout: int = 15,
    existing_hashes: Optional[Set[str]] = None
) -> Tuple[bool, Optional[str]]:
    """Download an image with validation and optional normalization.

    Args:
        url: Image URL to download.
        filename: Local filename to save to.
        max_size_mb: Maximum file size in MB.
        normalize: If > 0, resize to this size (square).
        timeout: Request timeout in seconds.
        existing_hashes: Set of existing image hashes for deduplication.

    Returns:
        Tuple of (success, image_hash or None).
    """
    max_bytes = int(max_size_mb * 1024 * 1024)
    if existing_hashes is None:
        existing_hashes = set()

    # Clean/resolve URL
    url = sanitize_google_url(url)
    if not url:
        print(f'❌ Skip non-image wrapper: {url}')
        return False, None
    if re.search(r'(wikipedia|wikimedia)\.org/.*/File:', url):
        resolved = resolve_wikimedia_file_page(url, timeout=timeout)
        if not resolved:
            print(f'❌ Could not resolve Wikimedia file page: {url}')
            return False, None
        url = resolved

    headers = {
        'User-Agent': random.choice(get_user_agents()),
        'Referer': 'https://www.google.com/',
        'Accept': 'image/*,*/*;q=0.8',
    }

    # HEAD preflight (content-type/length)
    ok, size, ctype = preflight_head(url, max_bytes, timeout=timeout)
    if not ok:
        human = f'{ctype or "non-image"}'
        if size and size > 0:
            human += f', {size/1024/1024:.1f}MB'
        print(f'❌ Skipping {filename}: HEAD says {human}')
        return False, None

    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()

            buf, downloaded = BytesIO(), 0
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    print(f'❌ Skipping {filename}: > {max_size_mb}MB while downloading')
                    return False, None
                buf.write(chunk)

        data = buf.getvalue()
        if not data or len(data) < 64:
            print(f'❌ Invalid image from {url}: empty/too small ({len(data)} bytes)')
            return False, None

        buf.seek(0)
        try:
            with Image.open(buf) as im:
                im.load()
                im = ImageOps.exif_transpose(im)
                if im.mode != 'RGB':
                    im = im.convert('RGB')

                # Calculate image hash for deduplication
                image_hash = calculate_image_hash(im)

                # Check for duplicates
                for existing_hash in existing_hashes:
                    if images_are_similar(image_hash, existing_hash):
                        print(f'❌ Skipping {filename}: similar image already exists')
                        return False, None

                if normalize and normalize > 0:
                    im = center_crop_and_resize(im, normalize)
                    print(f'✓ Downloaded & normalized: {filename} ({im.size[0]}x{im.size[1]})')
                else:
                    print(f'✓ Downloaded: {filename} ({im.size[0]}x{im.size[1]})')

                os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
                im.save(filename, format='JPEG', quality=90, optimize=True)
                return True, image_hash

        except UnidentifiedImageError as e:
            print(f'❌ Not an image (bytes were HTML/JSON/unsupported): {url} :: {e}')
            return False, None
        except OSError as e:
            print(f'❌ Corrupt/unsupported image from {url}: {e}')
            return False, None

    except requests.Timeout:
        print(f'❌ Failed to download {url}: timeout after {timeout}s')
        return False, None
    except requests.RequestException as e:
        print(f'❌ Failed to download {url}: {e}')
        return False, None
    except Exception as e:
        print(f'❌ Unexpected error for {url}: {e}')
        return False, None


def search_creative_commons_images(query: str, num_images: int = 10, retrieve: int = 5) -> List[str]:
    """Search Google Images for Creative Commons licensed images.

    Args:
        query: Search term.
        num_images: Target number of images to find.
        retrieve: Multiplier for initial search results.

    Returns:
        List of image URLs.
    """
    print(f"🔍 Searching for \'{query}\' with Creative Commons license...")
    search_query = quote(query)
    url = f'https://www.google.com/search?q={search_query}&tbm=isch&tbs=sur:fc'

    headers = {
        'User-Agent': random.choice(get_user_agents()),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        candidates = []

        # 1) img tags (often thumbnails, but some are direct)
        for img in soup.find_all('img'):
            src = img.get('data-src') or img.get('src') or ''
            if not src.startswith('http'):
                continue
            if 'gstatic' in src:  # skip Google cached thumbs
                continue
            candidates.append(src)

        # 2) URLs embedded in scripts
        for script in soup.find_all('script'):
            s = script.string or ''
            for m in re.findall(r'https://[^"]+\.(?:jpg|jpeg|png|gif|webp)', s, flags=re.IGNORECASE):
                if 'gstatic' in m:
                    continue
                candidates.append(m)

        # URL deduplication using set
        unique_candidates = list(dict.fromkeys(candidates))

        # Dedup and sanitize
        urls = []
        for u in unique_candidates:
            u = sanitize_google_url(u)
            if not u:
                continue
            # Resolve Wikimedia "File:" pages to binaries
            if re.search(r'(wikipedia|wikimedia)\.org/.*/File:', u):
                u = resolve_wikimedia_file_page(u) or ''
                if not u:
                    continue
            # Keep only likely images (extension); HEAD will further filter
            if is_direct_image_url(u) or 'upload.wikimedia.org' in u:
                urls.append(u)

        # Final URL deduplication
        urls = list(dict.fromkeys(urls))
        urls = urls[: max(num_images * retrieve, num_images)]
        print(f'✓ Found {len(urls)} potential direct image URLs')
        return urls[:num_images]

    except Exception as e:
        print(f'❌ Error searching images: {e}')
        return []


def search_bing_creative_commons(query: str, num_images: int = 10) -> List[str]:
    """Search Bing Images for Creative Commons licensed images.

    Args:
        query: Search term.
        num_images: Number of images to find.

    Returns:
        List of image URLs.
    """
    print(f"🔍 Searching Bing for \'{query}\' with Creative Commons license...")
    url = f'https://www.bing.com/images/search?q={quote(query)}&qft=+filterui:license-L2_L3_L4'

    headers = {
        'User-Agent': random.choice(get_user_agents()),
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

        # Prefer URLs with real extensions; dedup with dict.fromkeys()
        urls = [u for u in dict.fromkeys(candidates) if is_direct_image_url(u)]
        urls = urls[:num_images]
        print(f'✓ Found {len(urls)} Bing image URLs')
        return urls

    except Exception as e:
        print(f'❌ Error searching Bing: {e}')
        return []


def download_with_fallback(query: str, num_images: int = 10, normalize: int = -1) -> Optional[str]:
    """Download images with fallback and complete deduplication.

    Args:
        query: Search term.
        num_images: Number of images to download.
        normalize: If > 0, normalize images to this size.

    Returns:
        Path to download folder or None if no images downloaded.
    """
    print('🎯 Attempting Google Images first...')
    urls = search_creative_commons_images(query, num_images)

    if len(urls) < num_images // 2:  # If we get very few results
        print('⚡ Trying Bing as backup...')
        bing_urls = search_bing_creative_commons(query, num_images)
        urls.extend(bing_urls)
        # URL deduplication
        urls = list(dict.fromkeys(urls))

    if not urls:
        print('❌ No images found with either method')
        return None

    # Proceed with download
    folder = create_download_folder(query)
    downloaded_count = 0
    image_hashes: Set[str] = set()  # For RGB-level deduplication

    for i, url in enumerate(urls[:num_images * 2]):  # Try up to double the requested amount
        if downloaded_count >= num_images:
            break

        filename = os.path.join(folder, f'{query.replace(" ", "_")}_{downloaded_count + 1:02d}.jpg')

        success, img_hash = download_image(
            url, filename, normalize=normalize, existing_hashes=image_hashes
        )
        if success and img_hash:
            downloaded_count += 1
            image_hashes.add(img_hash)

        time.sleep(random.uniform(0.5, 1.5))

    print(f"\n🎉 Downloaded {downloaded_count} images to \'{folder}\' folder")
    return folder


def example_usage() -> None:
    """Show example usage of the script."""
    print('=' * 60)
    print('🖼️  CREATIVE COMMONS IMAGE DOWNLOADER')
    print('=' * 60)

    print('\nTo use this script:')
    print("1. Call download_creative_commons_images(\'your search term\', 10)")
    print("2. Or use download_with_fallback(\'your search term\', 10) for better results")


def interactive_download() -> None:
    """Interactive function for user input."""
    print('🎨 Creative Commons Image Downloader')
    print('-' * 40)

    query = input('Enter search term: ').strip()
    if not query:
        print('❌ Please enter a valid search term')
        return

    try:
        num_images = int(input('Number of images (default 10): ') or '10')
        if num_images < 1 or num_images > 50:
            print('❌ Please enter a number between 1 and 50')
            return
    except ValueError:
        num_images = 10

    # Download images
    folder = download_with_fallback(query, num_images)

    if folder:
        print(f'\n✨ Images saved to: {folder}')
        print('You can view them in the Colab file browser!')


if __name__ == '__main__':
    # Example usage
    example_usage()

    query = 'red'
    num_images = 10
    print(f"\n🔍 Searching for \'{query}\' with {num_images} images...")
    folder = download_with_fallback(query, num_images, normalize=256)

    if folder:
        print(f'Images downloaded to: {folder}')
    else:
        print('No images were downloaded.')

    # Uncomment to run interactive mode in Colab
    # interactive_download()
