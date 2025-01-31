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
python src/main.py --architecture <architecture> --model-path <model-path> --debug --config <config-file-path>
```
with an optional debug flag to see more detailed outputs.

Note that the config file should be in yaml format, and that any arguments you want to send to the huggingface API should be under the `model` key. See `configs/qwen_2b.yaml` as an example.

The supported architecture flags are currently:
- 'llava'
- 'qwen'

For example, one can run:
```base
python src/main.py --architecture qwen --model-path Qwen/Qwen2-VL-2B-Instruct --debug
```
or:
```base
python src/main.py --config configs/qwen_2b.yaml --debug
```
