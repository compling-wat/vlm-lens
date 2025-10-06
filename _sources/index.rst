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

+-----------------------------------------------+----------------------+
| Model (HuggingFace Identifier)                | Architecture Name    |
+===============================================+======================+
| CohereLabs/aya-vision-8b                      | aya-vision           |
+-----------------------------------------------+----------------------+
| Salesforce/blip2-opt-2.7b                     | blip2                |
+-----------------------------------------------+----------------------+
| Salesforce/blip2-opt-6.7b                     | blip2                |
+-----------------------------------------------+----------------------+
| Salesforce/blip2-opt-6.7b-coco                | blip2                |
+-----------------------------------------------+----------------------+
| openai/clip-vit-base-patch32                  | clip                 |
+-----------------------------------------------+----------------------+
| openai/clip-vit-large-patch14                 | clip                 |
+-----------------------------------------------+----------------------+
| THUDM/cogvlm-chat-hf                          | cogvlm               |
+-----------------------------------------------+----------------------+
| MBZUAI/GLaMM-FullScope                        | glamm                |
+-----------------------------------------------+----------------------+
| internlm/internlm-xcomposer2d5-7b             | internlm-xcomposer   |
+-----------------------------------------------+----------------------+
| OpenGVLab/InternVL2_5-1B                      | internvl             |
+-----------------------------------------------+----------------------+
| OpenGVLab/InternVL2_5-2B                      | internvl             |
+-----------------------------------------------+----------------------+
| OpenGVLab/InternVL2_5-4B                      | internvl             |
+-----------------------------------------------+----------------------+
| OpenGVLab/InternVL2_5-8B                      | internvl             |
+-----------------------------------------------+----------------------+
| deepseek-community/Janus-Pro-1B               | janus                |
+-----------------------------------------------+----------------------+
| deepseek-community/Janus-Pro-7B               | janus                |
+-----------------------------------------------+----------------------+
| llava-hf/bakLlava-v1-hf                       | llava                |
+-----------------------------------------------+----------------------+
| llava-hf/llava-1.5-7b-hf                      | llava                |
+-----------------------------------------------+----------------------+
| llava-hf/llava-1.5-13b-hf                     | llava                |
+-----------------------------------------------+----------------------+
| llava-hf/llama3-llava-next-8b-hf              | llavanext            |
+-----------------------------------------------+----------------------+
| llava-hf/llava-v1.6-mistral-7b-hf             | llavanext            |
+-----------------------------------------------+----------------------+
| llava-hf/llava-v1.6-vicuna-7b-hf              | llavanext            |
+-----------------------------------------------+----------------------+
| llava-hf/llava-v1.6-vicuna-13b-hf             | llavanext            |
+-----------------------------------------------+----------------------+
| openbmb/MiniCPM-o-2_6                         | minicpm              |
+-----------------------------------------------+----------------------+
| compling/MiniCPM-V-2                          | minicpm              |
+-----------------------------------------------+----------------------+
| allenai/Molmo-7B-D-0924                       | molmo                |
+-----------------------------------------------+----------------------+
| allenai/MolmoE-1B-0924                        | molmo                |
+-----------------------------------------------+----------------------+
| google/paligemma-3b-mix-224                   | paligemma            |
+-----------------------------------------------+----------------------+
| mistralai/Pixtral-12B-2409                    | pixtral              |
+-----------------------------------------------+----------------------+
| mistralai/Pixtral-12B-Base-2409               | pixtral              |
+-----------------------------------------------+----------------------+
| facebook/Perception-LM-1B                     | plm                  |
+-----------------------------------------------+----------------------+
| facebook/Perception-LM-3B                     | plm                  |
+-----------------------------------------------+----------------------+
| facebook/Perception-LM-8B                     | plm                  |
+-----------------------------------------------+----------------------+
| Qwen/Qwen2-VL-2B-Instruct                     | qwen                 |
+-----------------------------------------------+----------------------+
| Qwen/Qwen2-VL-7B-Instruct                     | qwen                 |
+-----------------------------------------------+----------------------+

Setup
-----

First, clone the repository:

.. code-block:: bash

   git clone https://github.com/compling-wat/vlm-lens.git
   cd vlm-lens

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
Replace ``<architecture>`` with the appropriate value (e.g., ``base``, ``cogvlm``):

.. code-block:: bash

   pip install -r envs/<architecture>/requirements.txt



Usage
-----

.. code-block:: bash

   python -m src.main --config <config-file-path>
