MiniCPM-o
================================


This tutorial guides you through extracting hidden representations for MiniCPM-o.


Dependency
-------------------------------
Environments are managed with `uv <https://docs.astral.sh/uv/>`_.
Create and activate the dedicated MiniCPM-o venv with the bundled switcher script:

.. code-block:: bash

   source scripts/use.sh minicpm-o

This creates ``.venvs/minicpm-o/`` on first use and installs the ``minicpm-o``
extra declared in ``pyproject.toml`` from the locked ``uv.lock``.

.. Note::

   MiniCPM-o supports two attention methods: ``sdpa`` and ``flash_attention_2`` (default).
   The default config uses ``sdpa``, so the ``minicpm-o`` extra does not include ``flash-attn``.
   If you want to use ``flash_attention_2``, install it separately after the sync, e.g.::

       uv pip install flash-attn==2.7.3 --no-build-isolation

   (see the inline comment in ``pyproject.toml`` for the pinned version).

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


To extract hidden representations on a CUDA-enabled device, execute:


.. code-block:: bash

   python -m src.main --config configs/minicpm-o.yaml --device cuda --debug

Results
-------------------------------

After successful execution, extracted layer outputs are saved as PyTorch tensors inside a SQL database file.
For the default config, the database is named ``minicpm-o.db``.
