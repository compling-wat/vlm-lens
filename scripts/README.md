# Helper Scripts
## read_tensor.py
Using the same `Config` object as before, we are trying to query for the specific prompts and embeddings saved from previously running the VLM extractor. In particular, by running:
```
python scripts/read_tensor.py --config configs/clip-base.yaml
```
one can read from the database for the specific configuration, including filtering on its model path, architecture, layers, prompts and image input.

## download_datasets.py
To download a Dataset with both text and images, run the following python file:
```
python download_datasets.py [yaml config file]
```

See existing config files in `configs/dataset/download-*.yaml` for examples. Notice that you still need to manually unzip the image after downloading. Notice that this is a relative path with respect to project root.

## map_datasets.py
If you want to map the text datasets to the image directory you downloaded locally using `download_datasets.py`, you can run the following command to map the datasets to image paths and store it as well locally. Make sure to provide a `save_path` location.
```
python map_datasets.py [yaml config file]
```
See existing config files like `configs/dataset/map-*.yaml` for reference. You need to provide the attribute names to be mapped. Otherwise, this functionality is done automatically within the model execution pipeline.

For the `image_regex` attribute, this is providing a pattern to convert the image ID column in the text dataset (`image_column`) into the corresponding image filenames saved under `image_dataset_path`, e.g., `123,456 -> 000000123456.jpg`. You may leave this blank if it doesn't require conversion.

## convert_images.py
If you need to convert the format of your input images before passing them, use the `convert_images.py` script like this:
```
python convert_images.py --format RGB --input_dir /path/to/input/images --output_dir /path/to/converted/images
```
