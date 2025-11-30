import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask
app = Flask(__name__)

# Folder to store uploaded images
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load trained model
MODEL_PATH = "densenet_ragi_model.keras"
model = load_model(MODEL_PATH)

# Load class labels
with open("class_labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Predict function
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = round(float(np.max(preds[0])) * 100, 2)

    predicted_label = class_labels[class_index]
    return predicted_label, confidence

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            predicted_label, confidence = predict_disease(filepath)

            return render_template(
                "index.html",
                file_path=filepath,
                predicted_label=predicted_label,
                confidence=confidence,
            )
    return render_template("index.html")

from flask import jsonify

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file:
        return jsonify({"error": "Empty file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    predicted_label, confidence = predict_disease(filepath)
    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
