# vlm-competence-dev

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
python src/main.py --architecture <architecture> --model-path <model-path> --debug --config <config-file-path> --input-dir <input-dir> --output-dir <output-dir>
```
with an optional debug flag to see more detailed outputs.

Note that the input and output directories default to the `./data` and `./output_dir` respectively.

Note that the config file should be in yaml format, and that any arguments you want to send to the huggingface API should be under the `model` key. See `configs/qwen-2b.yaml` as an example.

The supported architecture flags are currently:
- 'llava'
- 'qwen'
- 'clip'

For example, one can run:
```base
python src/main.py --architecture qwen --model-path Qwen/Qwen2-VL-2B-Instruct --debug
```
or:
```base
python src/main.py --config configs/qwen_2b.yaml --debug
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

### Prompt Input
For prompt input, one can either run using a single prompt over the entire input directory, entering a string under `text_input` or enter the following configuration:
```
text_prompts:
  - <prompt_1_name>
    - prompt: <prompt>
    - filter: <regex representing which image filenames to match to>
  - <prompt_2_name>
    - prompt: <prompt>
    - filter: <regex representing which image filenames to match to>
  ...
```

Note that filter will run within the input directory specified, and if no input directory exists, it defaults to `./data`.
