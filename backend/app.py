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
# Configure static folder - try React build first, fallback to simple HTML
react_build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build'))
simple_static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

print(f"🔍 React build path: {react_build_path}")
print(f"🔍 Simple static path: {simple_static_path}")
print(f"🔍 React build exists: {os.path.exists(react_build_path)}")
print(f"🔍 Simple static exists: {os.path.exists(simple_static_path)}")

# List contents of potential React build directory
if os.path.exists(react_build_path):
    print(f"🔍 React build contents: {os.listdir(react_build_path)}")
else:
    print("🔍 React build directory does not exist")

# List contents of parent directory to see what's available
parent_dir = os.path.dirname(react_build_path)
if os.path.exists(parent_dir):
    print(f"🔍 Parent directory contents: {os.listdir(parent_dir)}")

# Use React build if it exists, otherwise use simple static files
if os.path.exists(react_build_path) and os.path.exists(os.path.join(react_build_path, 'index.html')):
    static_folder = react_build_path
    print("✅ Using React build for frontend")
else:
    static_folder = simple_static_path
    print("✅ Using simple HTML for frontend")

print(f"🔍 Final static folder: {static_folder}")

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
            print("🔍 Current working directory:", os.getcwd())
            print("🔍 Backend directory:", os.path.dirname(__file__))
            
            # List files in backend directory
            backend_dir = os.path.dirname(__file__)
            if os.path.exists(backend_dir):
                print("🔍 Backend directory contents:", os.listdir(backend_dir))
            
            # Create classifier (will use trained model for Render)
            classifier = TransferLearningLatteArtClassifier()
            print(f"✅ Classifier initialized with classes: {classifier.class_names}")
            print(f"🔍 Model loaded: {classifier.model is not None}")
            if classifier.model is not None:
                print(f"🔍 Model type: {type(classifier.model)}")
            else:
                print("❌ No model loaded!")
        except Exception as e:
            print(f"❌ Error initializing classifier: {e}")
            import traceback
            print(f"❌ Full error: {traceback.format_exc()}")
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
            if classifier is None:
                return jsonify({'error': 'Classifier not available'}), 500
            
            print(f"🔍 Classifying image with model: {type(classifier.model)}")
            art_type, confidence = classifier.predict(image)
            print(f"🎯 Prediction: {art_type} with {confidence:.2f} confidence")
            
            # Prepare response (compatible with both frontends)
            response = {
                'art_type': art_type,
                'predicted_class': art_type,  # For React frontend compatibility
                'confidence': confidence
            }
            
            return jsonify(response)
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    classifier = get_classifier()
    model_status = "loaded" if classifier and classifier.model is not None else "not loaded"
    model_type = str(type(classifier.model)) if classifier and classifier.model else "none"
    
    # Check if model files exist
    backend_dir = os.path.dirname(__file__)
    model_files = []
    if os.path.exists(backend_dir):
        for file in os.listdir(backend_dir):
            if file.endswith('.h5'):
                model_files.append({
                    'name': file,
                    'size': os.path.getsize(os.path.join(backend_dir, file)),
                    'exists': True
                })
    
    return jsonify({
        'status': 'healthy', 
        'message': 'Latte Art Classifier is running',
        'model_status': model_status,
        'model_type': model_type,
        'classes': classifier.class_names if classifier else [],
        'model_files': model_files,
        'backend_dir': backend_dir,
        'working_dir': os.getcwd()
    })

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


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve the React app for all non-API routes"""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Use Render's PORT environment variable or default to 5001 for local development
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
