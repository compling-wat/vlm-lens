# vlm-competence-dev
This repository requires Python version 3.10.12+.

## Precommit Setup
We use Google docstring format for our docstrings and the pre-commit library to check our code. To install pre-commit, run the following command:

```bash
conda install pre-commit  # or pip install pre-commit
pre-commit install
```

The pre-commit hooks will run automatically when you try to commit changes to the repository.

## VLM Competence Embedding Extraction Script
To run the embedding extraction script, first download the dependencies through:
```bash
pip install -r requirements.txt
```

Then, execute the following command:
```bash
python src/main.py --architecture <architecture> --model-path <model-path> --debug --config <config-file-path> --input-dir <input-dir> --output-db <output-db>
```
with an optional debug flag to see more detailed outputs.

Note that the config file should be in yaml format, and that any arguments you want to send to the huggingface API should be under the `model` key. See `configs/qwen-2b.yaml` as an example.

The supported architecture flags are currently:
- 'llava'
- 'qwen'
- 'clip'
- 'glamm'
- 'janus'

For example, one can run:
```base
python src/main.py --architecture qwen --model-path Qwen/Qwen2-VL-2B-Instruct --debug
```
or:
```base
python src/main.py --config configs/qwen-2b.yaml --debug
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
#### Janus
For Janus, one needs to clone the separate submodules, which can be done with the following command:
```
git submodule update --recursive --init
```

## Running with different filters
We also provide a separate script that relies on the main functionality to run on multiple different filters, which can override the specific layer, prompt and input image directories, defaulting to the original layer, prompt and input image directories. This is specified through configs with the `-fc` or `--filter-config` flags as:
```
python src/main.py --config configs/clip-base.yaml --filter-config configs/clip-base-filter.yaml --debug
```
