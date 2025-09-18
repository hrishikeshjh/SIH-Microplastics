# Create API of ML model using flask

# Import libraries
import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.layers import TFSMLayer

# Create upload file path
upload_folder = "static/upload"

# Create Flask app
app = Flask(__name__, static_url_path='/static')
app.config['upload_folder'] = upload_folder

# Load the model
model = TFSMLayer("model/saved_model", call_endpoint="serving_default")

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Preprocess uploaded image
def preprocess_input_for_prediction(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        if 'exp' not in request.files:
            return 'No File!'

        file = request.files['exp']
        path = os.path.join(app.config['upload_folder'], file.filename)
        file.save(path)

        # Preprocess image
        input_data = preprocess_input_for_prediction(path)

        # Run prediction
        predictions = model(input_data)

        # Handle TFSMLayer output
        if isinstance(predictions, dict):
            if len(predictions) == 1:
                pred_array = list(predictions.values())[0]
            elif 'output_0' in predictions:
                pred_array = predictions['output_0']
            elif 'dense_1' in predictions:
                pred_array = predictions['dense_1']
            elif 'predictions' in predictions:
                pred_array = predictions['predictions']
            else:
                pred_array = list(predictions.values())[0]
        else:
            pred_array = predictions

        # Convert to numpy if it's a tensor
        if hasattr(pred_array, 'numpy'):
            pred_array = pred_array.numpy()

        # Flatten for easier handling
        pred_array = pred_array.flatten()

        # Binary prediction
        binary_predictions = (pred_array > 0.5).astype(int)

        # Assign labels (adjust if needed)
        labels = ['Microplastics', 'Clean']  # <-- Make sure order matches your model's output
        predictions_dict = {label: float(prob) for label, prob in zip(labels, pred_array)}

        # Determine top class
        class_predict = labels[np.argmax(pred_array)]

        return render_template(
            "results.html",
            image_path=path,
            class_predict=class_predict,
            binary_predictions="Yes" if binary_predictions[0] else "No",
            predictions=predictions_dict  # ✅ Now a dict for Jinja2
        )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
