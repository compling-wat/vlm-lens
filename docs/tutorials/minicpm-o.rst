MiniCPM-o
================================

This tutorial guides you through extracting MiniCPM-o model layers.

Dependency
-------------------------------
First, create and activate a virtual environment using `conda`:

.. code-block:: bash

   conda create -n <env_name> python=3.10
   conda activate <env_name>

Next, install the required dependencies via pip:

.. code-block:: bash

   pip install -r envs/minicpm/requirements.txt

.. Note::

   MiniCPM-o supports two attention methods: ``sdpa`` and ``flash_attention_2`` (default).
   The default config uses ``sdpa``, so ``requirements.txt`` does not include ``flash-attn``.
   If you want to use ``flash_attention_2``, be sure to install the corresponding package separately.

Configuration
-------------------------------
The main configuration file for MiniCPM-o is located at ``configs/minicpm-o.yaml``.
Refer to :ref:`Config Format <config_format>` for detailed explanation of all config options.
  
You can specify which modules or layers to register hooks for extraction.
A comprehensive list of available modules is provided in the log file: ``logs/openbmb/MiniCPM-o-2_6.txt``.

.. Note::

   For ``minicpm`` architecture implementation, we use a chat interface with a tokenizer limit set to 1 token to ensure exactly one forward pass.

Usage
-------------------------------

To run a forward pass and extract layers on a CUDA-enabled device, execute:

.. code-block:: bash

   python src/main.py --config configs/minicpm-o.yaml --device cuda

Results
-------------------------------

After successful execution, extracted layer outputs are saved as PyTorch tensors inside a SQL database file.
For the default config, the database is named ``minicpm-o.db``.

You can retrieve these tensors using the script ``scripts/read_tensor.py``, which lets you load and analyze the extracted data as needed.


