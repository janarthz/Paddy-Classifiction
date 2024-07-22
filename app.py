from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = load_model('D:\\Paddy Disease Project\\paddy_disease_classifier.h5')

# Define your class names and recovery tips
class_names = ['bacterial_leaf_streak', 'bacterial_leaf_blight', 'bacterial_panicle_blight', 'blast', 'brown-spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
recovery_tips = {
    'bacterial_leaf_streak': 'Use resistant varieties and apply appropriate bactericides.',
    'bacterial_leaf_blight': 'Use resistant varieties and ensure proper field sanitation.',
    'bacterial_panicle_blight': 'Avoid excessive nitrogen and use resistant varieties.',
    'blast': 'Apply fungicides and use resistant varieties.',
    'brown-spot': 'Improve soil fertility and apply fungicides.',
    'dead_heart': 'Use insecticides and remove affected plants.',
    'downy_mildew': 'Use resistant varieties and apply appropriate fungicides.',
    'hispa': 'Apply insecticides and use resistant varieties.',
    'normal': 'No disease detected. Maintain good agricultural practices.',
    'tungro': 'Use resistant varieties and control vector populations.'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class]

        image_url = url_for('uploaded_file', filename=filename)

        return render_template('result.html', image_url=image_url, prediction=predicted_class_name)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/tips/<disease>')
def tips(disease):
    tip = recovery_tips.get(disease, 'No recovery tips available for this disease.')
    return render_template('tips.html', disease=disease, tip=tip)

if __name__ == '__main__':
    app.run(debug=True)
