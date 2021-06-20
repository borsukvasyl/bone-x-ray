# Bone fracture localization

## Installation
Install code for model inference with `pip`
```shell
pip install git+https://github.com/borsukvasyl/bone-x-ray.git
```

## How to use
There are both classification and localization pretrained models available.
```python
import matplotlib.pyplot as plt
from skimage.io import imread
from bone_xray.classifier import Densenet121ClassificationPredictor
from bone_xray.localization import Densenet121LocalizationPredictor, visualize_heatmap

classifier = Densenet121ClassificationPredictor()
image = imread("image_path")
prediction = classifier(image)
print(f"Model prediction {prediction}")

localizer = Densenet121LocalizationPredictor()
cam = localizer(image)
visual = visualize_heatmap(image, cam)
plt.imshow(visual)
plt.show()
```

## Development setup
Local development setup with both training and inference code can be done with the following commands
```shell
pip install -r requirements.txt
python setup.py develop
```

Then, enter the following command from the project root to start default classification model training.
```shell
PYTHONPATH="." python model_training/classification/train.py model_training/classification/config/train.yaml
```

Training code for simple image classification network is located in `model_training/classification`.
Backbone architectures, models and training utilities are in `model_training/common`.

## User UI

User UI is implemented with streamlit.io

### To run with Docker

in Dockerfile change:

```
streamlit run --server.port $PORT app.py
```

to

```
streamlit run app.py
```
Then run:
```
docker build -t mystapp:latest .
docker run -p 8501:8501
```

### To run without Docker


```
pip3 install streamlit
streamlit run app.py
```
