Matching Layers
================================

To select layers to for extraction, we can use the Unix style strings, where ``*`` is used to denote wildcards.

For example, if we wanted to match with all the attention layer's query projection layer for Qwen, 
simply add the following lines to the config YAML file:

.. code-block:: bash
    
    modules:
        - model.layers.*.self_attn.q_proj
        