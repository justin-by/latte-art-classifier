# 🎨 Latte Art Classifier

A professional-grade machine learning application that classifies latte art into four categories: **heart**, **tulip**, **swan**, and **rosetta**.

## 🚀 Live Demo

Upload a photo of your latte art and get instant classification with confidence scores!

## 📋 Project Overview

This project demonstrates a complete machine learning pipeline:

1. **Data Acquisition**: Automated download from Kaggle using their API
2. **Model Training**: Transfer learning with MobileNetV2 for high accuracy
3. **Web Application**: React frontend + Flask backend
4. **Deployment Ready**: Configured for Render deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Flask Backend  │    │  Kaggle Dataset │
│                 │    │                 │    │                 │
│ • Image Upload  │◄──►│ • API Endpoints │◄──►│ • 460 Images    │
│ • Results Display│    │ • ML Model      │    │ • 4 Categories  │
│ • Modern UI     │    │ • Lazy Loading  │    │ • Professional  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Performance

- **Heart Detection**: 99%+ accuracy
- **Swan Detection**: 64%+ accuracy  
- **Tulip/Rosetta**: Good performance with some expected confusion
- **Model Size**: 23MB (MobileNetV2 transfer learning)
- **Response Time**: <2 seconds per classification

## 🛠️ Technology Stack

### Backend
- **Python 3.9+**
- **Flask** - Web framework
- **TensorFlow/Keras** - Machine learning
- **MobileNetV2** - Pre-trained model for transfer learning
- **Kaggle API** - Automated data acquisition

### Frontend
- **React 18** - User interface
- **Axios** - HTTP client
- **React Dropzone** - File upload
- **React Icons** - UI icons

## 📁 Project Structure

```
latte-art-classifier/
├── backend/
│   ├── app.py                          # Flask application
│   ├── transfer_learning_model.py      # ML model implementation
│   ├── kaggle_latte_art_model.h5       # Trained model (23MB)
│   ├── static/                         # Simple HTML fallback
│   └── requirements.txt                # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.js          # File upload component
│   │   │   ├── ResultsDisplay.js       # Results display component
│   │   │   └── Header.js               # Header component
│   │   └── App.js                      # Main React component
│   ├── public/                         # Static assets
│   └── package.json                    # Node.js dependencies
├── Dockerfile                          # Docker configuration
├── render.yaml                         # Render deployment config
├── start.sh                            # Development startup script
├── .gitignore                          # Git ignore rules
└── RENDER_DEPLOYMENT.md                # Deployment instructions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Kaggle API key (for data download)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd latte-art-classifier
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Build Frontend (Optional)
```bash
cd frontend
npm run build
```

### 5. Start Development Server
```bash
# From project root
bash start.sh
```

Visit `http://localhost:3000` to see the application!

## 📈 How It Works

### Step 1: Pre-trained Model
- Uses a pre-trained model trained on 460 professional latte art images
- Model trained with transfer learning using MobileNetV2
- Pre-trained on ImageNet for robust feature extraction
- Custom classification head for latte art categories

### Step 2: Classification Pipeline
1. User uploads image via React frontend
2. Flask backend receives image
3. Model preprocesses image (resize, normalize)
4. MobileNetV2 extracts features
5. Custom classifier predicts category
6. Returns result with confidence score

### Step 3: User Experience
- Drag-and-drop image upload
- Real-time classification results
- Confidence scores and visual feedback
- Tips for better latte art

## 🎯 Key Features

- **Pre-trained Model**: 460 high-quality images from Kaggle dataset
- **Transfer Learning**: Leverages pre-trained MobileNetV2
- **Lazy Loading**: Model loads only when needed (faster startup)
- **Modern UI**: Clean, responsive React interface with animations
- **API-First**: RESTful endpoints for easy integration
- **Production Ready**: Configured for Render deployment with Docker

## 🔧 API Endpoints

### `POST /api/classify`
Classify a latte art image.

**Request:**
- `multipart/form-data` with `image` field

**Response:**
```json
{
  "predicted_class": "heart",
  "confidence": 0.9919
}
```

### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Latte Art Classifier is running",
  "model_status": "loaded",
  "classes": ["heart", "tulip", "swan", "rosetta"]
}
```

## 🚀 Deployment

See `RENDER_DEPLOYMENT.md` for detailed deployment instructions to Render.

### Environment Variables (Render)
- No environment variables required - model is pre-trained and included

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Kaggle Dataset**: [Latte Art Train Dataset](https://www.kaggle.com/datasets/mingchenadam/latte-art-train)
- **TensorFlow**: For the machine learning framework
- **React**: For the frontend framework
- **Flask**: For the backend framework

---

**Built with ❤️ for coffee lovers and machine learning enthusiasts**