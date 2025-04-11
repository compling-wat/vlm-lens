# Test for new model classes

This test folder covers the suggested tests.

## Model class checking list

Before submitting a model, better to check if the following items are done.

Features

- [ ] requirement file: remember to push into the requirement PR
- [ ] model config yaml
- [ ] model architecture file
- [ ] the `run()` function

Suggested Tests (by running `pytest`)
- [ ] hidden_states are vectors of floats
- [ ] to input the same image/text twice and the output hidden_states are the same
- [ ] to input two different image/text and the output hidden_states are different
- [ ] bulk test on 10 images

- [ ] (curretnly uncovered) input_ids are vectors of integers
- [ ] (curretnly uncovered) work on image only inputs
- [ ] (curretnly uncovered) work on text only inputs

Details
- [ ] to set `model.eval()` (if needed)


## How to use the test model

1. Install the requirements

    also remember to install the requirements

    ```bash
    pip install pytest pytest-cov
    ```

2. In `test/test.py`, search for three `NOTE` signs and add your configuration as indicated.

3. Run

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
    pytest test/test.py -k 'test_hidden_states_diff_inputs'
    ```


## Notes

1. This test class currently does not support running in parallel, since the database has a lock protection.

2. If you want to add more tests, the test function should start with `test_` to be detected by pytest.
