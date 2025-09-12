import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaCloudUploadAlt, FaTimes } from 'react-icons/fa';
import axios from 'axios';
import './ImageUpload.css';

/**
 * ImageUpload Component
 * Handles drag-and-drop image upload and sends to AI classification API
 */
const ImageUpload = ({ onAnalysisStart, onAnalysisComplete, onAnalysisError, loading }) => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  /**
   * Handles file drop from drag-and-drop interface
   * @param {Array} acceptedFiles - Files dropped by user
   */
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedImage(file);
      const url = URL.createObjectURL(file); // Create preview URL
      setPreviewUrl(url);
    }
  }, []);

  // Configure drag-and-drop zone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif'] // Only accept image files
    },
    multiple: false // Only allow single file upload
  });

  /**
   * Sends image to backend AI classification API
   */
  const handleAnalyze = async () => {
    if (!uploadedImage) return;

    onAnalysisStart(); // Show loading state

    const formData = new FormData();
    formData.append('image', uploadedImage);

    try {
      // Send image to Flask backend for AI classification
      // Use relative URL for production, localhost for development
      const apiUrl = process.env.NODE_ENV === 'production' 
        ? '/api/classify' 
        : 'http://localhost:5001/api/classify';
      
      const response = await axios.post(apiUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      onAnalysisComplete(response.data); // Display results
    } catch (error) {
      console.error('Error analyzing image:', error);
      const errorMessage = error.response?.data?.error || 'Failed to analyze image. Please try again.';
      onAnalysisError(errorMessage);
    }
  };

  /**
   * Clears uploaded image and cleans up preview URL
   */
  const handleClear = () => {
    setUploadedImage(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl); // Free memory
      setPreviewUrl(null);
    }
  };

  return (
    <div className="image-upload-container fade-in">
      <div className="upload-section">
        {!uploadedImage ? (
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="upload-icon-container">
              <FaCloudUploadAlt className="upload-icon" />
            </div>
            <h3>Upload Your Latte Art</h3>
            <p>
              {isDragActive
                ? 'Drop the image here...'
                : 'Drag & drop an image here, or click to select'}
            </p>
            <p className="file-types">Supported formats: JPG, PNG, GIF</p>
            <div className="upload-hint">
              <span>✨ AI-powered classification</span>
            </div>
          </div>
        ) : (
          <div className="image-preview-container">
            <div className="image-preview">
              <img src={previewUrl} alt="Uploaded latte art" />
              <button className="clear-button" onClick={handleClear}>
                <FaTimes />
              </button>
            </div>
            <div className="image-info">
              <p><strong>File:</strong> {uploadedImage.name}</p>
              <p><strong>Size:</strong> {(uploadedImage.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >
              <span>{loading ? 'Analyzing...' : 'Analyze Latte Art'}</span>
              <div className="button-shine"></div>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;
