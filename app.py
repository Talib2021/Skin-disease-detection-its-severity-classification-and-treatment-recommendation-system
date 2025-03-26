from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from PIL import Image
import pillow_avif  # Enables AVIF support in PIL
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = [
    "Eczema", "Warts, Molluscum, and other Viral Infections", "Melanoma",
    "Atopic Dermatitis", "Basal Cell Carcinoma (BCC)", "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)", "Psoriasis, Lichen Planus, and related diseases",
    "Seborrheic Keratoses and other Benign Tumors", "Tinea Ringworm, Candidiasis, and other Fungal Infections"
]

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image preprocessing function
def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")  # Convert to RGB to handle all formats
    except Exception as e:
        print(f"Error loading image: {e}")
        return None  # Return None to handle errors properly

    img = img.resize((224, 224))  # Resize to match model input size
    img = np.array(img).astype("float32") #/ 255.0
    # img = preprocess_input(img) 
    img = np.expand_dims(img, axis=0)  # Expand for batch processing
    # print(img)
    return img

# Predict function
def model_predict(img_path, model):
    processed_img = preprocess_image(img_path)
    if processed_img is None:
        return "Invalid Image Format", 0.0  # Handle unsupported formats gracefully

    predictions = model.predict(processed_img)
    print( predictions)
    predicted_class = np.argmax(predictions)  # Get class with max probability
    confidence = np.max(predictions) * 100  # Confidence score

    return CLASS_LABELS[predicted_class], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    file_path = None  # Reset file path to avoid duplicates

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file!")

        if file:
            # Generate a unique filename using timestamp
            filename = f"{int(time.time())}_{file.filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Get prediction
            prediction, confidence = model_predict(file_path, model)

    return render_template("index.html", file_path=file_path, prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run()  # Removed debug=True to prevent auto-reloading issues
