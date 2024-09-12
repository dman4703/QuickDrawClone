import tensorflow as tf
import numpy as np

# Load a pre-trained model
model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')

def preprocess_image(image):
    # Resize the image to 224x224 pixels (for MobileNetV2)
    image = cv2.resize(image, (224, 224))
    image = np.stack((image,)*3, axis=-1)  # Convert grayscale to 3-channel image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def predict_drawing(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions[0][0][1]  # Return the highest predicted class
