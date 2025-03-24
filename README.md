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
For prompt input, one can either run using a single prompt over the entire input directory, entering a string under `prompt` or enter the following configuration:
```
text_prompts:
  - <prompt_1_name>
    - prompt: <prompt>
    - filter: <regex representing which image filenames to match to>
    - input_dir: <input directory to filter over>
  ...
```

Note that filter will run within the input directory specified, and if no input directory exists, it defaults to the high level input directory. If no prompt exists, it defaults to the high level input prompt and if no filter exists, it matches across all images.

If `text_prompts` is not specified, then it simply runs the prompt over the input directory for all images.

So for example,
```
prompt: "Describe the image."
input_dir: ./data
text_prompts:
  - filter_1:
    - prompt: "Describe this image in two words."
  - filter_2:
    - filter: red.*
    - input_dir: ./data_2
  - filter_3:
    - input_dir: ./data_3
```
will result in filter_1 being applied to the entirety of the input directory with the prompt "Describe the image in two words.". Then, the filter_2 with the prompt "Describe the image." will be applied over all `./data_2/red.*` files. Finally, filter_3 will be applied over all of `./data_3` with the prompt "Describe the image".

Note that if you would wish to test out multiple text prompts, one can do:
```
text_prompts:
  - filter_1:
    - prompt: "Describe this image in one word."
  - filter_2:
    - prompt: "Describe this image in two words."
  - filter_3:
    - prompt: "Describe this image in three words."
```
without specifying the default input directory.
