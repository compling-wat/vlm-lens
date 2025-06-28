.. _config_format:
Config YAML Format
================================

Using a configuration file lets you manage complex settings, version changes,
and avoid long command-line arguments.

Available fields are:

- **architecture**: Name of the model architecture to use (e.g., ``minicpm``).
- **model_path**: HuggingFace model identifier for the pre-trained model.
- **model**: List of model-specific args while initializing the model (e.g., ``torch_dtype``).
- **forward**: List of model-specific args while doing a forward pass (e.g., ``max_new_tokens``).
- **output_db**: Database name where results will be saved.
- **input_dir**: Directory containing the input data.
- **prompt**: Prompt for the model.
- **modules**: List of model layers to extract.


Example:

.. code-block:: yaml

    architecture: minicpm
    model_path: openbmb/MiniCPM-o-2_6
    model:
        - torch_dtype: auto
        - trust_remote_code: True
    forward:
        - max_new_tokens: 1
    output_db: minicpm-o.db
    input_dir: ./data/
    prompt: "Describe the color in this image in one word."
    modules:
        - llm.lm_head
        - vpm.encoder.layers.26