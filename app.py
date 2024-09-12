from flask import Flask, render_template, request, jsonify
from model.predict import predict_drawing
import base64
import re
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['image']
    # Convert base64 image data to numpy array
    image_data = re.sub('^data:image/.+;base64,', '', data_url)
    image = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    # Resize and predict the image using ML model
    result = predict_drawing(image)
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
