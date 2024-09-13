import cv2
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import load_model
import os

# List of classes (should be the same as in model_training.py)
classes = ['cat', 'dog', 'apple', 'banana', 'car', 'airplane', 'alarm clock', 'bathtub', 'bed', 'door',
           'dishwasher', 'helicopter', 'headphones', 'fence', 'garden hose', 'microwave', 'pizza', 'sun', 'whale']

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'DoodleVision.keras')
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    # Invert colors if background is black
    if np.mean(image) > 127:
        image = 255 - image
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    # Reshape image for model input
    image = image.reshape(1, 28, 28, 1)
    return image

def predict_drawing(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = classes[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])  # Convert to native float
    return predicted_class, confidence

if __name__ == "__main__":
    # Example usage:
    # Load an image from file
    image = cv2.imread('test_doodle.png', cv2.IMREAD_GRAYSCALE)
    predicted_class, confidence = predict_drawing(image)
    print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
