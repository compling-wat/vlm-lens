CogVLM
================================

This tutorial guides you through extracting hidden representations for CogVLM.

Dependencies
-------------------------------
Create a virtual environment using ``conda``:

.. code-block:: bash

    conda create -n <env_name> python=3.11
    conda activate <env_name>

Install the required dependencies via ``pip``:

.. code-block:: bash

   pip install -r envs/cogvlm/requirements.txt

Configurations
-------------------------------
The default configuration file for CogVLM is located at ``configs/cogvlm-chat.yaml``.
Refer to :ref:`Config YAML Format <config_format>` for detailed explanation of general configuration options.

The following are specific config fields for CogVLM:

- ``template_version``: Version type for CogVLM to use. Available options are ``base``, ``chat``, and ``vqa``.
- ``max_new_tokens``: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

.. Note::

    We recommend adhering to the default values for other CogVLM specific config fields
    (e.g. ``trust_remote_code``, ``tokenizer_path``) as per the
    `HuggingFace quickstart tutorial for CogVLM
    <https://huggingface.co/THUDM/cogvlm-chat-hf#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B%EF%BC%88qiuckstart%EF%BC%89>`_.

To view which modules (or layers) are available for representation extraction,
a comprehensive list of modules for CogVLM is provided in the log file ``logs/THUDM/cogvlm-chat-hf.txt``.

General Usage
-------------------------------
To extract hidden representations for CogVLM on a CUDA-enabled device with the default config file,
run the following:

.. code-block:: bash

   python src/main.py --config configs/cogvlm-chat.yaml --device cuda --debug

.. Note::

   CogVLM requires at least 40GB of VRAM for inference.

Outputs
-------------------------------
Extracted tensor outputs are saved as PyTorch tensors inside a SQL database file.
In the default config file, the output database is ``cogvlm.db``.

Additional Links
-------------------------------
| `CogVLM: Visual Expert for Pretrained Language Models <https://arxiv.org/abs/2311.03079>`_
| `THUDM/cogvlm-chat-hf <https://huggingface.co/THUDM/cogvlm-chat-hf>`_
