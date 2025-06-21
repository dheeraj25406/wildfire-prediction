import numpy as np
from tensorflow.keras.models import load_model as keras_load
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

CLASS_NAMES = ['No Wildfire', 'Wildfire']

def load_model(path):
    try:
        model = keras_load(path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model {path}: {str(e)}")

def preprocess_image(image, model_name):
    if model_name == "VGG16":
        image = image.resize((224, 224))
    else:
        image = image.resize((128, 128))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_class(model, image):
    try:
        pred = model.predict(image)[0][0]
        prediction = CLASS_NAMES[int(pred > 0.5)]
        probability = pred if pred > 0.5 else 1 - pred
        return prediction, probability
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
