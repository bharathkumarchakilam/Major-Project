from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load models
models = {
    'FCN': load_model('models/depression_detection_fcn.h5'),
    'FFL': load_model('models/depression_detection_ffl.h5'),
    'LSTM': load_model('models/depression_lstm_model.h5'),
    'HYBRID': load_model('models/depression_detection_hybrid_model.h5')
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'model_type' not in request.form:
        flash('Model type not selected')
        return redirect(request.url)

    model_type = request.form['model_type']
    file = request.files.get('file')

    if file and allowed_file(file.filename) and model_type in models:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        face_paths = crop_faces(filepath)

        if not face_paths:
            return render_template('result.html', predictions=[('No face detected', None)], model_name=model_type)

        predictions = []
        for face_path in face_paths:
            result = predict(face_path, model_type)
            predictions.append((result, os.path.basename(face_path)))

        return render_template('result.html', predictions=predictions, model_name=model_type)

    flash('Invalid file or model selection')
    return redirect(request.url)

def crop_faces(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return []

    cropped_paths = []
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        unique_name = f"face_{uuid.uuid4().hex[:8]}.jpg"
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        face_image.save(face_path)
        cropped_paths.append(face_path)

    return cropped_paths

def predict(filepath, model_type):
    img = Image.open(filepath)

    input_shapes = {
        'FCN': (128, 128, 3),
        'FFL': (128, 128, 3),
        'LSTM': (48, 48, 1),
        'HYBRID': (128, 128, 3)
    }

    target_size, channels = input_shapes[model_type][:2], input_shapes[model_type][2]
    img = img.resize(target_size)

    if model_type == 'LSTM':
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img_array = np.array(img)

    if model_type == 'LSTM':
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = models[model_type]
    prediction = model.predict(img_array)
    predicted_value = float(prediction[0][0])

    return 'Depressed 😢' if predicted_value < 0.5 else 'Not Depressed 😊'

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
