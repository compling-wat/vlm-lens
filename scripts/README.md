# Helper Scripts

## download_datasets.py
To download a Dataset with both text and images, run the following python file:
```
python download_datasets.py [yaml config file]
```

See existing config files in `configs/dataset/download-*.yaml` for examples. Notice that you still need to manually unzip the image after downloading. Notice that this is a relative path with respect to project root.

## convert_images.py
If you need to convert the format of your input images before passing them, use the `convert_images.py` script like this:
```
python convert_images.py --format RGB --input_dir /path/to/input/images --output_dir /path/to/converted/images
```
