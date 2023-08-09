# Robot-Assisted Vocabulary Learning System Using Pure Synthetic Data

This repository contains resources associated with our journal article titled "Development of A Novel Robot-Assisted Vocabulary Learning System Using Pure Synthetic Data". Specifically, you will find model configuration files, trained model weights, and our synthetic datasets.

## Directory Structure

- `card_det_models_configs`: Contains the model configuration files necessary to reproduce our training and evaluation.
- `card_det_models_weights`: Contains a script to download the trained model checkpoints. (Note: We are currently preparing to upload the model weights, please check back soon.)
- `synthetic_datasets`: Contains a script to download the synthetic card datasets. (Note: We are currently preparing to upload the datasets, please check back soon.)

## How to Run Training and Evaluation using mmdetection

### Prerequisites

Before you begin, ensure you have `mmdetection` installed. If not, you can follow the installation instructions from [mmdetection's official repository](https://github.com/open-mmlab/mmdetection).

### Training

1. Navigate to the `card_det_models_configs` directory.
2. Use the following command to initiate training:

```bash
python tools/train.py <path_to_config_file>
```
Replace `<path_to_config_file>` with the path to the desired model config file.

### Evaluation

1. Ensure that you've downloaded the trained model weights using the script in `card_det_models_weights`.
2. Navigate to the `card_det_models_configs` directory.
3. Use the following command for evaluation:

```bash
python tools/test.py <path_to_config_file> <path_to_checkpoint> --eval bbox
```
Replace `<path_to_config_file>` with the path to the model config file, and `<path_to_checkpoint>` with the path to the downloaded weights.

## Note on Model Weights and Datasets

We are in the process of preparing and uploading the model weights and datasets. We appreciate your patience and encourage you to check back soon for updates.
