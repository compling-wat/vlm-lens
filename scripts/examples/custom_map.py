"""Example script to define a custom mapping function for map_datasets.py."""


def custom_map_fn(img_id: int) -> str:
    """Custom map function to map dataset entries to converted CLEVR filenames.

    Args:
        img_id (int): The image ID to convert.

    Returns:
        str: The converted image filename.
    """
    clean_id = str(img_id).replace(',', '').replace('.png', '')
    return clean_id + '.jpg'
