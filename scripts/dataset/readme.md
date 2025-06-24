## download_datasets.py
To download a Dataset with both text and images, run the following python file:
```
python download_datasets.py [yaml config file]
```

See existing config files like "configs/dataset/clevr.yaml" for examples. Notice that you still need to manually unzip the image after downloading. Notice that this is a relative path with respect to project root.

## map_datasets.py
If you want to map the text datasets to the image directory you downloaded locally using `download_datasets.py`, you can run the following command to map the datasets to image paths and store it as well locally. Make sure to provide a `save_path` location.
```
python map_datasets.py [yaml config file]
```
See existing config files like `configs/*-dataset.yaml` for reference. You need to provide the attribute names to be mapped. Otherwise, this functionality is done automatically within the model execution pipeline.
