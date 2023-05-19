from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tensorflow import keras

from PIL import Image


def preprocess_image(image_file):
    image = Image.open(image_file)
    np_image = np.array(image)
    resize = keras.layers.Resizing(224, 224)

    np_image = np.array(resize(np_image)).astype(int)
    np_image = np_image[np.newaxis, ...]

    return np_image


class TrainedModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, trained_model):
        self.trained_model = trained_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply your trained model for predictions
        predictions = self.trained_model.predict(X, verbose=False)
        return np.argmax(predictions)
