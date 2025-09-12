from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
from transfer_learning_model import TransferLearningLatteArtClassifier

# Create Flask app
# Try multiple possible locations for the frontend build directory
possible_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build')),  # Local development
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')),  # Heroku build
    os.path.abspath('frontend/build'),  # Alternative path
    os.path.abspath('build'),  # Direct build path
]

static_folder = None
for path in possible_paths:
    print(f"🔍 Checking path: {path}")
    if os.path.exists(path):
        print(f"✅ Found static folder at: {path}")
        print(f"🔍 Contents: {os.listdir(path)}")
        static_folder = path
        break

if static_folder is None:
    print("❌ No static folder found!")
    static_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build'))

app = Flask(__name__, static_folder=static_folder, static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize classifier (load model lazily)
classifier = None

def get_classifier():
    """Get or initialize the classifier"""
    global classifier
    if classifier is None:
        try:
            print("🔄 Initializing classifier...")
            # Create classifier without loading default model
            classifier = TransferLearningLatteArtClassifier()
            # Override the default model loading
            classifier.model = None
            print("🔄 Loading Kaggle model file...")
            classifier.load_model('kaggle_latte_art_model.h5')
            print(f"✅ Model loaded with classes: {classifier.class_names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            classifier = None
    return classifier

@app.route('/api/classify', methods=['POST'])
def classify_latte_art():
    """Endpoint to classify and grade latte art"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Read and process the image
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Classify the latte art
            classifier = get_classifier()
            art_type, confidence = classifier.predict(image)
            
            # Prepare response
            response = {
                'art_type': art_type,
                'confidence': confidence
            }
            
            return jsonify(response)
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Latte Art Classifier is running'})

@app.route('/api/status', methods=['GET'])
def status_check():
    """Status check endpoint"""
    global classifier
    model_loaded = classifier is not None
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'message': 'Server is running' + (' with model loaded' if model_loaded else ' (model will load on first request)')
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check file structure"""
    import os
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    frontend_dir = os.path.join(current_dir, 'frontend')
    
    debug_info = {
        'current_dir': current_dir,
        'parent_dir': parent_dir,
        'frontend_dir': frontend_dir,
        'frontend_dir_exists': os.path.exists(frontend_dir),
        'static_folder': app.static_folder,
        'static_folder_exists': os.path.exists(app.static_folder) if app.static_folder else False,
        'current_dir_contents': os.listdir(current_dir) if os.path.exists(current_dir) else [],
        'parent_dir_contents': os.listdir(parent_dir) if os.path.exists(parent_dir) else [],
    }
    
    if os.path.exists(frontend_dir):
        debug_info['frontend_dir_contents'] = os.listdir(frontend_dir)
    
    if app.static_folder and os.path.exists(app.static_folder):
        debug_info['static_folder_contents'] = os.listdir(app.static_folder)
    
    return jsonify(debug_info)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve the React app for all non-API routes"""
    print(f"🔍 Serving path: '{path}'")
    print(f"🔍 Static folder: {app.static_folder}")
    print(f"🔍 Static folder exists: {os.path.exists(app.static_folder)}")
    
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        print(f"🔍 Serving file: {path}")
        return send_from_directory(app.static_folder, path)
    else:
        print(f"🔍 Serving index.html")
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Use Heroku's PORT environment variable or default to 5001 for local development
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
