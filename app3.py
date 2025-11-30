import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# ==========================
# 1. Initialize Flask App
# ==========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ==========================
# 2. Load Model & Labels
# ==========================
MODEL_PATH = "tiny_dataset_model.keras"   # your MobileNetV2 model
LABELS_PATH = "class_labels.txt"          # label file

model = load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

print("✅ Model and labels loaded successfully!")

# ==========================
# 3. Prediction Function
# ==========================
def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Expand dims (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    
    predicted_label = class_labels[pred_index]
    confidence = round(float(np.max(preds[0]) * 100), 2)

    return predicted_label, confidence

# ==========================
# 4. Routes
# ==========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Predict
    label, confidence = predict_image(file_path)

    return render_template(
        'index.html',
        prediction=label,
        confidence=confidence,
        img_path=file_path
    )

# ==========================
# 5. Run Flask App
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
