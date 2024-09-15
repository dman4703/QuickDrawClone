from flask import Flask, render_template, request, jsonify
from model.predict import predict_drawing
import base64
import re
import numpy as np
import cv2
import os
import subprocess
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Directory to save user drawings for retraining
RETRAIN_DIR = os.path.join('model', 'retrain_data')  # Updated path

if not os.path.exists(RETRAIN_DIR):
    os.makedirs(RETRAIN_DIR)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    svg_data = request.json.get('svg')
    if not svg_data:
        return jsonify({'error': 'No SVG data provided'}), 400

    # Save SVG to a temporary file
    svg_filename = 'temp_drawing.svg'
    png_filename = 'temp_drawing.png'
    try:
        with open(svg_filename, 'w') as svg_file:
            svg_file.write(svg_data)

        # Use Inkscape to convert SVG to PNG
        inkscape_command = [
            'C:\\Program Files\\Inkscape\\bin\\inkscape.exe',  # Ensure this is the correct path
            '--export-type=png',
            '--export-filename=' + png_filename,
            svg_filename
        ]
        subprocess.run(inkscape_command, check=True)

        # Load the PNG image and resize it to 28x28
        image = Image.open(png_filename).convert('L')
        image = image.resize((28, 28))  # Resize to 28x28
        image = np.array(image)

        # Proceed with prediction
        predicted_class, confidence = predict_drawing(image)

        # Clean up temporary files
        os.remove(svg_filename)
        os.remove(png_filename)

        return jsonify({'prediction': predicted_class, 'confidence': confidence})

    except FileNotFoundError as e:
        return jsonify({'error': 'Inkscape command failed. Ensure Inkscape is installed and the path is correct.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.json
    if not data or 'svg' not in data:
        return jsonify({'error': 'No SVG data provided'}), 400

    label = data.get('label', 'unknown')  # default to 'unknown' if label is missing
    svg_data = data['svg']

    # Save SVG to a temporary file
    svg_filename = 'temp_retrain_drawing.svg'
    png_filename = f'temp_retrain_{label}.png'
    try:
        with open(svg_filename, 'w') as svg_file:
            svg_file.write(svg_data)

        # Use Inkscape to convert SVG to PNG
        inkscape_command = [
            'C:\\Program Files\\Inkscape\\bin\\inkscape.exe',  # Ensure this is the correct path
            '--export-type=png',
            '--export-filename=' + png_filename,
            svg_filename
        ]
        subprocess.run(inkscape_command, check=True)

        # Load the PNG image
        image = Image.open(png_filename).convert('L')
        image = np.array(image)

        # Save the image to the retrain directory under the correct label
        label_dir = os.path.join(RETRAIN_DIR, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image_count = len(os.listdir(label_dir))
        image_filename = f"{label}_{image_count + 1}.png"
        image_path = os.path.join(label_dir, image_filename)
        Image.fromarray(image).save(image_path)

        # Clean up temporary files
        os.remove(svg_filename)
        os.remove(png_filename)

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
