# Test for new model classes


## Model class checking list

Before submitting a model, better to check if the following items are done.


Features

- [ ] requirement file: remember to push into the requirement PR
- [ ] model config yaml
- [ ] model architecture file
- [ ] generate input ids (if needed)
- [ ] forward (if needed)

Suggested Tests
- [ ] input_ids are vectors of integers
- [ ] hidden_states are vectors of floats
- [ ] to input the same image/text twice and the output hidden_states are the same
- [ ] to input two different image/text and the output hidden_states are different
- [ ] to input images of different sizes and the input_ids have the same shape (not applied to some models)

- [ ] work on image only inputs (priority low)
- [ ] work on text only inputs (priority low)
- [ ] stress test

Details
- [ ] to set `model.eval()` (if needed)
- [ ] load model configs should be in the yaml file instead of the code


## How to use the test model

Install the requirements

also remember to install the requirements

```bash
pip install pytest pytest-cov
```

run all tests

```bash
pytest -s test/test.py
```

run a subset of tests

```bash
pytest test/test.py -k '<test_name>'
```

e.g.,

```bash
pytest -s test/test.py -k 'test_hidden_states_diff_inputs'
```
