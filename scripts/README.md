# Helper Scripts
## read_tensor.py
Using the same `Config` object as before, we are trying to query for the specific prompts and embeddings saved from previously running the VLM extractor. In particular, by running:
```
python scripts/read_tensor.py --config configs/clip-base.yaml
```
one can read from the database for the specific configuration, including filtering on its model path, architecture, layers, prompts and image input.

## read_filter_tensor.py
Given a filter config, simply provide this just as the running script with:
```
python scripts/read_filter_tensor.py --config configs/clip-base.yaml --filter-config configs/clip-base-filter.yaml
```
where it will read out the specific architecture, layers, prompts and image input.
## print_layer_names.py
A utility for inspecting PyTorch model checkpoints hosted on Hugging Face Hub.
**Example:**
```
python print_layer_names.py --project zhangtao-whu/OMG-LLaVA -pth omg_llava_7b_finetune_8gpus.pth --limit 0 --show-shapes
```
Please read the description at the top of the python file for a more detailed description.
