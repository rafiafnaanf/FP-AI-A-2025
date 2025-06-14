import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from deepface import DeepFace
import logging
import cv2

# Configure basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created upload folder at {UPLOAD_FOLDER}")

AVAILABLE_MODELS = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
]
AVAILABLE_METRICS = ["cosine", "euclidean", "euclidean_l2"]
AVAILABLE_BACKENDS = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface'
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
        return jsonify({'error': 'Missing one or both image files'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    # Get form data
    model_name = request.form.get('model_name', 'VGG-Face')
    distance_metric = request.form.get('distance_metric', 'cosine')
    detector_backend = request.form.get('detector_backend', 'opencv')
    anti_spoofing_str = request.form.get('anti_spoofing', 'false')
    anti_spoofing = anti_spoofing_str.lower() in ['true', 'on']


    if model_name not in AVAILABLE_MODELS: return jsonify({'error': 'Invalid model selected.'}), 400
    if distance_metric not in AVAILABLE_METRICS: return jsonify({'error': 'Invalid distance metric selected.'}), 400
    if detector_backend not in AVAILABLE_BACKENDS: return jsonify({'error': 'Invalid backend detector selected.'}), 400
    if file1.filename == '' or file2.filename == '': return jsonify({'error': 'No selected file for one or both images'}), 400
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)): return jsonify({'error': 'Invalid file type.'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    try:
        file1.save(filepath1)
        file2.save(filepath2)
        
        img1 = cv2.imread(filepath1)
        img2 = cv2.imread(filepath2)

        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images. They may be corrupted or in an unsupported format.")

        logging.info(f"Verification options: model={model_name}, metric={distance_metric}, backend={detector_backend}, anti-spoofing={anti_spoofing}")

        if anti_spoofing:
            img1_faces = DeepFace.extract_faces(img_path=img1, detector_backend=detector_backend, enforce_detection=True, anti_spoofing=True)
            if len(img1_faces) > 1: raise ValueError("Multiple faces detected in Image 1. Please use an image with only one face.")
            if not img1_faces[0]['is_real']: raise ValueError("Spoof detected in Image 1.")
            
            img2_faces = DeepFace.extract_faces(img_path=img2, detector_backend=detector_backend, enforce_detection=True, anti_spoofing=True)
            if len(img2_faces) > 1: raise ValueError("Multiple faces detected in Image 2. Please use an image with only one face.")
            if not img2_faces[0]['is_real']: raise ValueError("Spoof detected in Image 2.")

            result = DeepFace.verify(img1_path=img1_faces[0]['face'], img2_path=img2_faces[0]['face'], model_name=model_name, distance_metric=distance_metric, enforce_detection=False)
        else:
            result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name=model_name, distance_metric=distance_metric, detector_backend=detector_backend, enforce_detection=True)
        
        return jsonify({
            'verified': result.get('verified'),
            'distance': round(result.get('distance', 0.0), 4),
            'threshold': result.get('threshold', 0.0),
            'model': result.get('model'),
            'similarity_metric': result.get('similarity_metric')
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error with backend {detector_backend}: {error_message}")

        if "Face could not be detected" in error_message:
            error_msg = f"Could not detect a face using the '{detector_backend}' backend. This can happen with low-quality or obscured images. Please try a clearer picture or a different backend like 'opencv'."
            return jsonify({'error': error_msg}), 400
        
        if detector_backend in ['retinaface', 'mtcnn']:
            error_msg = f"The '{detector_backend}' backend can be slow on first use or may fail on certain images. Please try again, or select a different backend. Details: {error_message}"
            return jsonify({'error': error_msg}), 500
            
        return jsonify({'error': f'An unexpected error occurred: {error_message}'}), 500
    finally:
        if os.path.exists(filepath1): os.remove(filepath1)
        if os.path.exists(filepath2): os.remove(filepath2)

if __name__ == '__main__':
    app.run(debug=True)
