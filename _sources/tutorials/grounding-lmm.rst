Grounding-LMM
================================


This tutorial guides you through extracting hidden representations for Glamm (Grounding-LMM).

We provide an Apptainer configuration. See the readme at envs/glamm/readme.md for details. Here we only show the normal approach.


Dependency
-------------------------------
This model relies on `groundingLMM` submodule. We use our own version at https://github.com/compling-wat/groundingLMM.

First, load the submodule by running

.. code-block:: bash

   git submodule update --init src/models/glamm/groundingLMM

in the root directory.

Next, create and activate a virtual environment using ``conda``:

.. code-block:: bash

   conda create -n <env_name> python=3.10
   conda activate <env_name>

Then, install the required dependencies via pip (must be done in order):

.. code-block:: bash

   pip install packaging
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   pip install -r envs/glamm/requirements.txt

Then we need to install mmcv from source

.. code-block:: bash

   git clone https://github.com/open-mmlab/mmcv
   cd mmcv
   git checkout v1.4.7
   MMCV_WITH_OPS=1 pip install -e .
   pip install jmespath

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

   python -m src.main --config configs/glamm.yaml --device cuda --debug

Results
-------------------------------

After successful execution, extracted layer outputs are saved as PyTorch tensors inside a SQL database file.
For the default config, the database is named ``glamm.db``.
