import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ResultsDisplay from './components/ResultsDisplay';
import Header from './components/Header';

/**
 * Main App Component
 * Manages state for image upload, AI analysis, and results display
 */
function App() {
  // State management for the application
  const [results, setResults] = useState(null); // AI classification results
  const [loading, setLoading] = useState(false); // Loading state during analysis
  const [error, setError] = useState(null); // Error messages

  /**
   * Handles successful AI analysis completion
   * @param {Object} data - AI classification results from backend
   */
  const handleAnalysisComplete = (data) => {
    setResults(data);
    setLoading(false);
    setError(null);
  };

  /**
   * Handles AI analysis errors
   * @param {string} errorMessage - Error message to display
   */
  const handleAnalysisError = (errorMessage) => {
    setError(errorMessage);
    setLoading(false);
    setResults(null);
  };

  /**
   * Initiates AI analysis process
   */
  const handleAnalysisStart = () => {
    setLoading(true);
    setError(null);
    setResults(null);
  };

  /**
   * SCROLL ANIMATIONS - Adds fade-in effects as elements come into view
   * Uses Intersection Observer API for performance
   */
  useEffect(() => {
    const observerOptions = {
      threshold: 0.1, // Trigger when 10% of element is visible
      rootMargin: '0px 0px -50px 0px' // Start animation 50px before element enters viewport
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-in'); // Add animation class
        }
      });
    }, observerOptions);

    // Observe all elements with fade-in class
    const elements = document.querySelectorAll('.fade-in');
    elements.forEach(el => observer.observe(el));

    return () => observer.disconnect(); // Cleanup observer
  }, [results]); // Re-run when results change (new analysis)

  return (
    <div className="App">
      {/* HEADER - App title and branding */}
      <Header />
      
      {/* HERO SECTION - Main landing area with examples */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title fade-in">
            AI-Powered Latte Art Classifier
          </h1>
          <p className="hero-subtitle fade-in">
            Upload your latte art and discover what pattern you've created with our advanced machine learning model
          </p>
          
          {/* LATTE ART EXAMPLES - Visual preview of supported patterns */}
          <div className="latte-examples fade-in">
            <div className="example-item">
              <div className="example-icon heart">❤️</div>
              <span>Heart</span>
            </div>
            <div className="example-item">
              <div className="example-icon tulip">🌷</div>
              <span>Tulip</span>
            </div>
            <div className="example-item">
              <div className="example-icon swan">🦢</div>
              <span>Swan</span>
            </div>
            <div className="example-item">
              <div className="example-icon rosetta">🌹</div>
              <span>Rosetta</span>
            </div>
          </div>
        </div>
      </section>

      {/* MAIN CONTENT AREA */}
      <main className="main-content">
        <div className="container">
          {/* IMAGE UPLOAD COMPONENT - Drag & drop interface */}
          <ImageUpload 
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={handleAnalysisError}
            loading={loading}
          />
          
          {/* LOADING STATE - Shows during AI analysis */}
          {loading && (
            <div className="loading-container fade-in">
              <div className="loading-spinner"></div>
              <p>Analyzing your latte art...</p>
            </div>
          )}
          
          {/* ERROR STATE - Shows if analysis fails */}
          {error && (
            <div className="error-container fade-in">
              <p className="error-message">{error}</p>
            </div>
          )}
          
          {/* RESULTS DISPLAY - Shows AI classification results */}
          {results && <ResultsDisplay results={results} />}
        </div>
      </main>
    </div>
  );
}

export default App;
