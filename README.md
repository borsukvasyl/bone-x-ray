# Bone fracture localization

## Project structure
Training code for simple image classification network is located in `model_training/classification`.
Backbone architectures, models and training utilities are in `model_training/common`.

## Prerequisites

- Python 3.6
- `pip install -r requirements.txt`

## Train classification model
Enter the following command from the project root
```shell
PYTHONPATH="." python model_training/classification/train.py model_training/classification/config/train.yaml
```
