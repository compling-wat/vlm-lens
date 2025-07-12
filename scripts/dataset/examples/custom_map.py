"""Example script to define a custom mapping function for map_datasets.py."""


def custom_map_fn(img_id) -> str:
    """Custom map function to map dataset entries to converted CLEVR filenames."""
    clean_id = str(img_id).replace(',', '').replace('.png', '')
    return clean_id + '.jpg'
