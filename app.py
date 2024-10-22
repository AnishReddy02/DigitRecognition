from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = load_model('mnist_model.h5')

# Ensure 'uploads' folder exists for saving uploaded images
os.makedirs('uploads', exist_ok=True)

# Function to preprocess the uploaded image
def preprocess_image(image):
    """Convert uploaded image to a format suitable for the model."""
    try:
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
        print(f"Image shape: {image_array.shape}")  # Debug shape
        print(f"Sample pixel values: {image_array[0, :5, :5, 0]}")  # Debug values
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']

        if file.filename == '':
            return "No file selected", 400

        try:
            # Open and preprocess the image
            image = Image.open(file)
            
            # Save uploaded image for debugging
            image.save(os.path.join('uploads', file.filename))
            print(f"Uploaded image saved as: uploads/{file.filename}")

            processed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)

            # Log prediction probabilities and label
            print(f"Prediction probabilities: {prediction}")
            print(f"Predicted label: {predicted_label}")

            # Render the template with the prediction result
            return render_template('index.html', prediction=predicted_label)

        except Exception as e:
            print(f"Error processing file: {e}")
            return "An error occurred while processing the image.", 500

    # Render the template for GET requests
    return render_template('index.html')

# Test route to validate the model's predictions
@app.route('/test', methods=['GET'])
def test_prediction():
    # Generate random input similar to the MNIST dataset
    test_input = np.random.rand(1, 28, 28, 1)
    prediction = model.predict(test_input)
    predicted_label = np.argmax(prediction)
    return f"Test Prediction: {predicted_label}"

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
