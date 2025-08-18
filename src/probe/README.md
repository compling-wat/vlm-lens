# Probe Implementation
Using the output database of extracted features (generated using `src/main.py`), you can easily intialize probes using the script in `probe/main.py`.

To do this, you need a probe config `yaml` file specifying: the probe `model` attributes, `training` and `test` configuration and the input `data`. The following is an example of a probe configuration file for features extracted using LLAVA-1.5-7b:
```yaml
model:
  - activation: ReLU # a valid activation function from torch.nn
  - hidden_size: 512 # the input and output size of the intermediate layers
  - num_layers: 2 # the number of layers of the probe model
  - save_dir: /path/to/save_dir # the location to save the probe results

training:
  - batch_size: [64, 128, 1024]
  - num_epochs: [50, 100, 200]
  - learning_rate: [0.001, 0.0005, 0.0001]
  - optimizer: AdamW # a valid optimizer from torch.nn
  - loss: CrossEntropyLoss # a valid loss metric from torch.nn

test:
  - batch_size: 32
  - loss: CrossEntropyLoss # a valid loss metric from torch.nn

data:
  - input_db: /path/to/input_db
  - db_name: tensors # the name of the database in input_db, the default is `tensors` if unspecified
  - input_layer: language_model.model.layers.16.post_attention_layernorm # the layer value in input_db to filter by

```

> For `data`, the `input_layer` value must be specified as the probe model is only trained on data extracted from a single layer to avoid dimension mismatches or ambiguous results.

## Training
Note that the training procedure conducts an iterative (naive) hyperparameter search on all configurations of the `training` hyperparameters: `batch_size`, `num_epochs` and `learning_rate`. It uses $k$-fold cross validation (with a default of $k=5$) to store the lowest validation loss from each configuration and returns the configuration with the minimum validation loss.

```yaml
training:
  - batch_size: [64, 128, 1024]
  - num_epochs: [50, 100, 200]
  - learning_rate: [0.001, 0.0005, 0.0001]
  ...
```

## Testing
After retrieving the best training configuration, the current script trains and tests a probe model on two versions of the data: a **Main** condition (where the data is untouched) and a **Shuffled** condition (where the labels are shuffled).

This is based on work from [Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/), where they propose the inclusion of "control tasks" to ensure that the probe is selective in its learning. The goal is to achieve high task-specific accuracy and low control task accuracy. The output attributes described in [Results](#results) follow the former experimental design.

## Results
Finally, the program saves both the probe model and result values in `save_dir`. The current script saves the attributes to a `{save_dir}/probe_data.txt`. The following is an example output:
```python
{
    # The final train configuration with the lowest validation loss
    "train_config": {
                    "batch_size": 64,
                    "num_epochs": 200,
                    "learning_rate": 0.0001,
                    "optimizer": "AdamW",
                    "loss": "CrossEntropyLoss"
                    },

    # Probe accuracy on the shuffled data
    "shuffle_accuracy": 0.3295668661594391,
    "shuffle_loss": 2.1123009968163153,
    "shuffle_preds": [1, 2, 0, ... ],
    "shuffle_labels": [0, 0, 0, ... ],

    # Probe accuracy on the original data
    "test_accuracy": 0.6436911225318909,
    "test_loss": 1.7786695135290789,
    "test_preds": [2, 0, 2, ...],
    "test_labels": [0, 0, 2, ...],

    # The statistical significance of the difference between the shuffled and unshuffled data using a z-test
    "pvalue": 4.8257723322914694e-116
}
```
Details on the reason behind shuffling and experimental design can be found in the [Testing](#testing) section.
