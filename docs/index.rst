VLM-Lens Documentation
================================

.. toctree::
   :maxdepth: 2
   :hidden:

   modules

Overview
-------------------------------
This repository provides utilities for extracting hidden states from
state-of-the-art vision-language models. By surfacing these intermediate
representations, you can perform a comprehensive analysis of the knowledge
encoded within each model.

.. _supported_models:

Supported Models
-------------------------------
We currently support extracting hidden states from the following vision-language models.
The architecture name (used for model selection) is shown in square brackets:

- **BLIP-2** [blip2]
- **CLIP** [clip]
- **CogVLM** [cogvlm]
- **Glamm** [glamm]
- **InternLM-XComposer** [internlm-xcomposer]
- **InternVL** [internvl]
- **Janus** [janus]
- **LLaVa** [llava]
- **MiniCPM-V2** [minicpm]
- **MiniCPM-o** [minicpm]
- **Molmo** [molmo]
- **OMG-LLaVa** [//PENDING//]
- **PaliGemma** [paligemma]
- **Qwen** [qwen]

Setup
-----

First, clone the repository:

.. code-block:: bash

   git clone https://github.com/repo/repo.git
   cd repo

Because each model may have different dependencies,
it is recommended to use a separate virtual environment for each model you run.

For example, using ``conda``:

.. code-block:: bash

   conda create -n <env_name> python=3.10
   conda activate <env_name>

Or, using native ``python venv``:

.. code-block:: bash

   python -m venv <env_name>
   source <env_name>/bin/activate

After activating your environment, install dependencies for your desired model architecture.
Replace ``<architecture>`` with the appropriate value (e.g., ``blip2``, ``llava``):

.. code-block:: bash

   pip install -r envs/<architecture>.requirements.txt



Usage
-----

.. code-block:: bash

   python -m src.main --architecture <architecture> --model-path <model-path> --debug --config <config-file-path> --input-dir <input-dir> --output-db <output-db>
