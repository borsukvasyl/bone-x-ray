import io
import streamlit as st
import numpy as np
from PIL import Image

from bone_xray.classifier import Densenet121ClassificationPredictor
from bone_xray.localization import Densenet121LocalizationPredictor, visualize_heatmap

TEXT_UPLOAD = "Upload bone x-rays to run a prediction"
TEXT_PROBABILITY = "{}: probability of abnormality is {}%"
TEXT_WRONG_FORMAT = "Wrong format of image"
TEXT_WAIT = "Please wait, running prediction for {}. It may take some time"
TEXT_DONE = 'All done'

PREDICTION_NEGATIVE = 0
PREDICTION_POSITIVE = 1

visualizer = Densenet121LocalizationPredictor()
classifier = Densenet121ClassificationPredictor()

if __name__ == '__main__':

    uploaded_files = st.file_uploader(TEXT_UPLOAD, accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()

        try:

            imageStream = io.BytesIO(bytes_data)
            img = Image.open(imageStream).convert('RGB')
            img = np.array(img)

            st.write(TEXT_WAIT.format(uploaded_file.name))

            probability = float(classifier(img))
            prediction = PREDICTION_POSITIVE if probability > 0.5 else PREDICTION_NEGATIVE

            cam = visualizer(img)
            visual = visualize_heatmap(img, cam)
            visual = np.hstack([img, visual])

            st.write(TEXT_PROBABILITY.format(uploaded_file.name, round(probability * 100, 2)))
            if prediction == PREDICTION_NEGATIVE:
                st.image(img)
            else:
                st.image(visual)

        except Exception:
            st.write(TEXT_WRONG_FORMAT)

    st.write(TEXT_DONE)
