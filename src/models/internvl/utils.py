"""Helper functions from official huggingface library of InternVL."""

from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448) -> T.Compose:
    """Helper function that transform image.

    Args:
        input_size (int, optional): The input size. Defaults to 448.

    Returns:
        T.Compose: The composed transform.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])


def find_closest_aspect_ratio(
        aspect_ratio: float, target_ratios: List[Tuple[float, float]],
        width: int, height: int, image_size: int) -> Tuple[int, int]:
    """Helper function that find closest aspect ratio.

    Args:
        aspect_ratio (float): The existing image aspect ratio.
        target_ratios (list): The target aspect ratios.
        width (int): The original image width.
        height (int): The original image height.
        image_size (int): The target image size.

    Returns:
        tuple: The closest aspect ratio.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
        image: Image, min_num: int = 1, max_num: int = 12,
        image_size: int = 448, use_thumbnail: bool = False) -> List[Image]:
    """Helper function.

    Args:
        image (Image): The input image.
        min_num (int, optional): The minimum number of image patches. Defaults to 1.
        max_num (int, optional): The maximum number of image patches. Defaults to 12.
        image_size (int, optional): The target image size. Defaults to 448.
        use_thumbnail (bool, optional): Whether to use thumbnail. Defaults to False.

    Returns:
        list: The processed images.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Load image to pixel values.

    Args:
        image_file (str): The image file path.
        input_size (int, optional): The input size. Defaults to 448.
        max_num (int, optional): The max number of image patches. Defaults to 12.

    Returns:
        torch.Tensor: The corresponding pixel values.
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
