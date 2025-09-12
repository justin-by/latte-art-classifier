# Latte Art Classifier Backend

This is the backend API for the Latte Art Classifier & Grader application. It provides endpoints for classifying and grading latte art images.

## Features

- **Image Classification**: Detects the type of latte art (heart, tulip, rosetta, etc.)
- **Grading System**: Scores symmetry, contrast, and sharpness
- **Feedback Generation**: Provides personalized feedback based on results
- **RESTful API**: Easy-to-use endpoints for frontend integration

## Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /api/classify
Upload and analyze a latte art image.

**Request**: Multipart form data with an image file
**Response**: JSON with classification and grading results

Example response:
```json
{
  "art_type": "heart",
  "confidence": 0.85,
  "grades": {
    "symmetry": 78.5,
    "contrast": 65.2,
    "sharpness": 72.1,
    "overall_score": 72.6
  },
  "feedback": [
    "Great attempt at a heart pattern!",
    "Excellent symmetry!",
    "Great contrast achieved!"
  ]
}
```

### GET /api/health
Health check endpoint.

**Response**: JSON with server status

## Technical Details

- **Framework**: Flask
- **Image Processing**: OpenCV
- **Machine Learning**: TensorFlow/Keras (model structure ready for training)
- **CORS**: Enabled for frontend integration

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF

## Grading Criteria

1. **Symmetry (40% weight)**: Measures how symmetrical the design is
2. **Contrast (30% weight)**: Measures the contrast between milk and coffee
3. **Sharpness (30% weight)**: Measures the sharpness and definition of edges

## Future Improvements

- Train the model with actual latte art dataset
- Add more sophisticated image preprocessing
- Implement user authentication and history
- Add support for video analysis
