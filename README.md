# <img src="imgs/logo.png" alt="VLM-Lens Logo" height="48" style="vertical-align:middle; margin-right:50px;"/> VLM-Lens

[![python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&style=flat-square)](https://www.python.org/downloads/release/python-31012/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)
[![Documentation](https://img.shields.io/badge/Documentation-Online-blue.svg?style=flat-square)](https://compling-wat.github.io/vlm-lens/)

## Table of Contents

- [Environment Setup](#environment-setup)
- [Example Usage: Extract Qwen2-VL-2B Embeddings with VLM-Lens](#example-usage-extract-qwen2-vl-2b-embeddings-with-vlm-lens)
  - [General Command-Line Demo](#general-command-line-demo)
  - [Run Qwen2-VL-2B Embeddings Extraction](#run-qwen2-vl-2b-embeddings-extraction)

## Environment Setup
We recommend using a virtual environment to manage your dependencies. You can create one using the following command to create a virtual environment under
```bash
virtualenv --no-download "venv/vlm-lens-base" --prompt "vlm-lens-base"
source venv/vlm-lens-base/bin/activate
```

Then, install the required dependencies:
```bash
pip install -r envs/base/requirements.txt
```

There are some models that require different dependencies, and we recommend creating a separate virtual environment for each of them to avoid conflicts.
For such models, we have offered a separate `requirements.txt` file under `envs/<model_name>/requirements.txt`, which can be installed in the same way as above.
All the model-specific environments are independent of the base environment, and can be installed individually.

**Notes**:
1. There may be local constraints (e.g., issues caused by cluster regulations) that cause failure of the above commands. In such cases, you are encouraged to modify it whenever fit. We welcome issues and pull requests to help us keep the dependencies up to date.
2. Some models, due to the resources available at the development time, may not be fully supported on modern GPUs. While our released environments are tested on L40s GPUs, we recommend following the error messages to adjust the environment setups for your specific hardware.

## Example Usage: Extract Qwen2-VL-2B Embeddings with VLM-Lens

### General Command-Line Demo

The general command to run the quick command-line demo is:
```bash
python src/main.py \
  --config <config-file-path> \
  --debug
```
with an optional debug flag to see more detailed outputs.

Note that the config file should be in yaml format, and that any arguments you want to send to the huggingface API should be under the `model` key.
See `configs/qwen-2b.yaml` as an example.

### Run Qwen2-VL-2B Embeddings Extraction
The file `configs/qwen-2b.yaml` contains the configuration for running the Qwen2-VL-2B model.

```yaml
architecture: qwen  # Architecture of the model, see more options in src/models/configs.py
model_path: Qwen/Qwen2-VL-2B-Instruct  # HuggingFace model path
model:  # Model configuration, i.e., arguments to pass to the model
  - torch_dtype: auto
output_db: output/qwen.db  # Output database file to store embeddings
input_dir: ./data/  # Directory containing images to process
prompt: "Describe the color in this image in one word."  # Textual prompt
modules:  # List of modules to extract embeddings from
  - lm_head
  - visual.blocks.31
```

To run the extraction on available GPU, use the following command:
```base
python src/main.py \
  --config configs/qwen-2b.yaml \
  --device cuda \
  --debug
```

### Matching Layers
To automatically set up which layers to find/use, one should use the Unix style strings, where you can use `*` to denote wildcards.

For example, if one wanted to match with all the attention layer's query projection layer for Qwen, simply add the following lines to the .yaml file:
```
modules:
  - model.layers.*.self_attn.q_proj
```

#### Printing out Named Modules
Unfortunately there is no way to find which layers to potentially match to without loading the model. This can take quite a bit of system time figuring out.

Instead, we offer some cached results under `logs/` for each model, which were generated through including the `-l` or `--log_named_modules` flag.

When running this flag, it is not necessary to set modules or anything besides the architecture and HuggingFace model path.

### Using a Cache
To use a specific cache, one should set the `HF_HOME` environment variable as so:
```
HF_HOME=./cache/ python src/main.py --config configs/clip-base.yaml --debug
```

### Output Database
Specified by the `-o` and `--output-db` flags, this specifies the specific output database we want. From this, in SQL we have a single table under the name `tensors` with the following columns:
```
Name, Architecture, Timestamp, Image Path, Layer, Tensor
```
where each column is:
1. `Name` represents the model path from HuggingFace.
2. `Architecture` is the supported flags above.
3. `Timestamp` is the specific time that the model was ran.
4. `Image path` is the absolute path to the image.
5. `Layer` is the matched layer from `model.named_modules()`
6. `Tensor` is the embedding saved.

### Using specific models
#### Glamm
For Glamm (GroundingLMM), one needs to clone the separate submodules, which can be done with the following command:
```
git submodule update --recursive --init
```


## Running with different filters
We also provide a separate script that relies on the main functionality to run on multiple different filters, which can override the specific layer, prompt and input image directories, defaulting to the original layer, prompt and input image directories. This is specified through configs with the `-fc` or `--filter-config` flags as:
```
python src/run_filters.py --config configs/clip-base.yaml --filter-config configs/clip-base-filter.yaml --debug
```
