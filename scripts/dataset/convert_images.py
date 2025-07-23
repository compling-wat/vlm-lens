"""Script to convert image formats."""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def convert_images(img_format: str, input_dir: str, output_dir: str) -> None:
    """Converts the images in input_dir to img_format, and saves it to output_dir.

    Args:
        img_format (str): The image format to convert the images
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where converted images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = Path(input_dir).glob('*')
    for img_path in tqdm(image_paths, desc='Converting images'):
        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert(img_format)
                out_path = Path(output_dir) / (img_path.stem + '.jpg')
                rgb_img.save(out_path, format='JPEG', quality=95)
        except Exception as e:
            print(f'Failed to convert {img_path}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert images to specific image format.')
    parser.add_argument('--format', type=str, default='RGB',
                        help='Format to convert the images to (e.g., RGB, RGBA).')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input image directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save converted images.')

    args = parser.parse_args()
    convert_images(args.format, args.input_dir, args.output_dir)
