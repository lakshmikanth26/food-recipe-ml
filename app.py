from flask import Flask, request, render_template
from tensorflow.keras.layers import DepthwiseConv2D as KerasDepthwiseConv2D
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

class CustomDepthwiseConv2D(KerasDepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Paths to model and data
MODEL_PATH = './model/food101_model.h5'
LABELS_PATH = './model/labels.txt'
RECIPES_PATH = './model/recipes.json'

# Load the trained model
model = load_model(MODEL_PATH, compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load labels from labels.txt
with open(LABELS_PATH, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Load recipes from recipes.json
with open(RECIPES_PATH, 'r') as file:
    recipes = json.load(file)

# Utility to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Prediction logic
def predict_image(filepath):
    """
    Preprocess the image, make predictions, and return the predicted class.
    """
    # Preprocess the image
    img = load_img(filepath, target_size=(224, 224))  # Adjust size as per your model
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions[0])]
    return predicted_class

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure folder exists
        file.save(filepath)

        # Predict the class
        predicted_class = predict_image(filepath)

        # Fetch the recipe for the predicted class
        recipe = recipes.get(predicted_class, f"Recipe for {predicted_class} is not available.")
        
        return render_template('result.html', label=predicted_class, recipe=recipe, image_url=filepath)

    return "Invalid file type", 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
