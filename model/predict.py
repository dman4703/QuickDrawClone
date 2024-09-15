import tensorflow as tf
import numpy as np
import os

# List of classes (should be the same as in model_training.py)
classes = ['cat', 'dog', 'apple', 'car', 'airplane', 'alarm clock', 'bathtub', 'bed', 'door',
           'dishwasher', 'helicopter', 'headphones', 'fence', 'garden hose', 'microwave', 'pizza', 'sun', 'whale']

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'DoodleVision.keras')
model = tf.keras.models.load_model(model_path)

def preprocess_image(image, invert=True):
    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0

    # Invert colors if necessary (ensure black strokes on white background)
    if invert and np.mean(image) > 0.5:
        image = 1.0 - image

    # Reshape image for model input
    image = image.reshape(1, 28, 28, 1)
    return image


def predict_drawing(image, invert=True):
    processed_image = preprocess_image(image, invert=invert)
    predictions = model.predict(processed_image)

    if len(predictions) == 0:
        return 'undefined', 0.0

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = classes[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])  # Convert to native float

    if predicted_class is None:
        predicted_class = 'undefined'

    print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
    return predicted_class, confidence


if __name__ == "__main__":
    # Example usage:
    # Assuming you have a PNG image rendered from SVG at 28x28 pixels
    from PIL import Image

    # Load the image (ensure it's in grayscale mode)
    image_path = 'test_doodle.png'  # Replace with your image path
    image = Image.open(image_path).convert('L')
    image = np.array(image)

    # Call predict_drawing with the invert parameter as needed
    predicted_class, confidence = predict_drawing(image, invert=True)
    print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
