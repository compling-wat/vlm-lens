Grounding-LMM
================================


This tutorial guides you through extracting hidden representations for Glamm (Grounding-LMM).


Dependency
-------------------------------
This model relies on `groundingLMM` submodule. We use our own version at https://github.com/compling-wat/groundingLMM.

First, load the submodule by running `git submodule update --init src/models/glamm/groundingLMM` in the root directory.

Next, create and activate a virtual environment using ``conda``:

.. code-block:: bash

   conda create -n <env_name> python=3.10
   conda activate <env_name>

Then, install the required dependencies via pip:

.. code-block:: bash

   pip install -r envs/glamm/requirements.txt

.. Note::

   This model enforces the using of cuda 11. If your system cuda version is not cuda 11, you can attempt to run `module load cuda/11` if you are on a slurm-based cluster system. If this doesn't work, install a virtual cuda environment inside the conda environment. Run `conda install -c conda-forge cudatoolkit-dev=11.7` before installing the `requirements.txt` mentioned before.

If you meet installation problem after following the tutorial above, feel free to create an issue in our repo.

Configuration
-------------------------------
The main configuration file for Glamm is located at ``configs/glamm.yaml``.
Refer to :ref:`Config Format <config_format>` for detailed explanation of all config options.

You can specify which modules or layers to register hooks for extraction.
A comprehensive list of available modules is provided in the log file: ``logs/MBZUAI/GLaMM-FullScope.txt``.

.. Note::

   For ``Glamm`` architecture implementation, we use a chat (evaluate) interface with a tokenizer limit set to 1 token to ensure exactly one forward pass.

Usage
-------------------------------


To extract hidden representations on a CUDA-enabled device, execute:


.. code-block:: bash

   python src/main.py --config configs/glamm.yaml --device cuda --debug

Results
-------------------------------

After successful execution, extracted layer outputs are saved as PyTorch tensors inside a SQL database file.
For the default config, the database is named ``glamm.db``.

You can retrieve these tensors using the script ``scripts/read_tensor.py``, which lets you load and analyze the extracted data as needed.
