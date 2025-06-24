List Model Layers
================================

If you want to inspect the internal layers or modules of a model,
our script provides a convenient way to log and examine the model's architecture.

Make sure the model you want to inspect is one of the :ref:`supported models <supported_models>`, 
and have its architecture name handy. 
Additionally, ensure all required dependencies are installed in your virtual environment.

To run the script, use the following command:

.. code-block:: bash

    python src/main.py -a <architecture> -m <model_path> -l

Here, ``<architecture>`` is the architecture name, and ``<model_path>`` is the HuggingFace model identifier.

After executing the script, you will find a text file listing the layer names at:

``logs/<namespace>/<model_name>.txt``
