# Bone fracture localization

## Project structure
Training code for simple image classification network is located in `model_training/classification`.
Backbone architectures, models and training utilities are in `model_training/common`.

## Installation
Install with `pip` code for model inference
```shell
pip install git+https://github.com/borsukvasyl/bone-x-ray.git
```

Local development setup with both training and inference code
```shell
pip install -r requirements
python setup.py develop
```

## Train classification model
Enter the following command from the project root
```shell
PYTHONPATH="." python model_training/classification/train.py model_training/classification/config/train.yaml
```
