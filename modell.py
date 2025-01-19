import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Charger le modèle MobileNetV2 pré-entrainé
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

# Prédiction à partir d'une image
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    print(f"Predictions for {image_path}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}. {label} - {score*100:.2f}%")
