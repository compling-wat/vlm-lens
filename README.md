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
python src/main.py --architecture <architecture> --model-path <model-path> --debug
```
with an optional debug flag to see more detailed outputs.

The supported architecture flags are currently:
- 'llava'
- 'qwen'

For example, one can run:
```base
python src/main.py --architecture qwen --model-path Qwen/Qwen2-VL-2B-Instruct --debug
```
