CogVLM
================================

This tutorial guides you through extracting CogVLM model layers.

Dependencies
-------------------------------
Create a virtual environment using conda:

.. code-block:: bash
    conda create -n <env_name> python=3.11
    conda activate <env_name>

Install the required dependencies via pip:

.. code-block:: bash
   pip install -r envs/cogvlm/requirements.txt

Configurations
-------------------------------
The default configuration file for CogVLM is located at ``configs/cogvlm-chat.yaml``.
Refer to :ref:`Config YAML Format <config_format>` for detailed explanation of general config options.