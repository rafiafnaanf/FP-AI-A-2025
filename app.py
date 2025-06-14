import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from deepface import DeepFace
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created upload folder at {UPLOAD_FOLDER}")

AVAILABLE_MODELS = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
]
AVAILABLE_METRICS = ["cosine", "euclidean", "euclidean_l2", "angular"]
AVAILABLE_BACKENDS = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface',
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        logging.error("One or both image files are missing from the request.")
        return jsonify({'error': 'Missing one or both image files'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    model_name = request.form.get('model_name', 'VGG-Face')
    distance_metric = request.form.get('distance_metric', 'cosine')
    detector_backend = request.form.get('detector_backend', 'opencv')
    anti_spoofing_str = request.form.get('anti_spoofing', 'false')
    anti_spoofing = anti_spoofing_str.lower() in ['true', 'on']


    if model_name not in AVAILABLE_MODELS:
        logging.error(f"Invalid model_name received: {model_name}")
        return jsonify({'error': f'Invalid model selected. Choose from: {", ".join(AVAILABLE_MODELS)}'}), 400
    if distance_metric not in AVAILABLE_METRICS:
        logging.error(f"Invalid distance_metric received: {distance_metric}")
        return jsonify({'error': f'Invalid distance metric selected. Choose from: {", ".join(AVAILABLE_METRICS)}'}), 400
    if detector_backend not in AVAILABLE_BACKENDS:
        logging.error(f"Invalid detector_backend received: {detector_backend}")
        return jsonify({'error': f'Invalid backend detector selected. Choose from: {", ".join(AVAILABLE_BACKENDS)}'}), 400

    if file1.filename == '' or file2.filename == '':
        logging.error("One or both image filenames are empty.")
        return jsonify({'error': 'No selected file for one or both images'}), 400

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        logging.error("One or both files have an unsupported extension.")
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)

    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    try:
        file1.save(filepath1)
        logging.info(f"Saved image1 to {filepath1}")
        file2.save(filepath2)
        logging.info(f"Saved image2 to {filepath2}")

        logging.info(f"Attempting verification with model: {model_name}, metric: {distance_metric}, backend: {detector_backend}, anti-spoofing: {anti_spoofing}")

        if anti_spoofing:
            img1 = cv2.imread(filepath1)
            img2 = cv2.imread(filepath2)

            face_objs1 = DeepFace.extract_faces(img_path=img1, detector_backend=detector_backend, enforce_detection=True, anti_spoofing=True)
            if not face_objs1 or not face_objs1[0]['is_real']:
                 raise ValueError("Spoof detected or no real face found in Image 1.")

            face_objs2 = DeepFace.extract_faces(img_path=img2, detector_backend=detector_backend, enforce_detection=True, anti_spoofing=True)
            if not face_objs2 or not face_objs2[0]['is_real']:
                 raise ValueError("Spoof detected or no real face found in Image 2.")

            result = DeepFace.verify(
                img1_path=face_objs1[0]['face'],
                img2_path=face_objs2[0]['face'],
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=False
            )

        else:
            result = DeepFace.verify(
                img1_path=filepath1,
                img2_path=filepath2,
                model_name=model_name,
                distance_metric=distance_metric,
                detector_backend=detector_backend,
                enforce_detection=True
            )

        logging.info(f"DeepFace verification result: {result}")

        return jsonify({
            'verified': result.get('verified'),
            'distance': round(result.get('distance', 0.0), 4),
            'threshold': result.get('threshold', 0.0),
            'model': result.get('model'),
            'similarity_metric': result.get('similarity_metric')
        })

    except ValueError as e:
        error_message = str(e)
        logging.error(f"Validation Error: {error_message}")
        return jsonify({'error': error_message}), 400

    except Exception as e:
        error_message = str(e)
        logging.error(f"Error during DeepFace verification (Model: {model_name}, Metric: {distance_metric}): {error_message}")
        if "Face could not be detected" in error_message or "cannot be aligned" in error_message :
            return jsonify({'error': f'Could not detect a face in one or both images using {detector_backend}. Please use clearer images. Details: {error_message}'}), 400
        elif "not found" in error_message and any(ext in error_message for ext in [".h5", ".pb", ".pkl", ".json"]):
            return jsonify({'error': f'Model files for {model_name} might be missing or corrupted. DeepFace might need to redownload them. Details: {error_message}'}), 500
        else:
            return jsonify({'error': f'An unexpected error occurred: {error_message}'}), 500
    finally:
        if os.path.exists(filepath1):
            os.remove(filepath1)
            logging.info(f"Removed image1 from {filepath1}")
        if os.path.exists(filepath2):
            os.remove(filepath2)
            logging.info(f"Removed image2 from {filepath2}")

if __name__ == '__main__':
    app.run(debug=True)
