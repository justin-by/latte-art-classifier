import React from 'react';
import { FaHeart, FaStar } from 'react-icons/fa';
import './ResultsDisplay.css';

/**
 * ResultsDisplay Component
 * Displays the AI classification results with confidence score and tips
 */
const ResultsDisplay = ({ results }) => {
  const { art_type, confidence } = results;

  /**
   * Returns the appropriate icon for each latte art type
   * @param {string} type - The detected latte art type
   * @returns {JSX.Element} - React icon component
   */
  const getArtTypeIcon = (type) => {
    switch (type?.toLowerCase()) {
      case 'heart':
        return <FaHeart className="art-type-icon heart" />;
      case 'tulip':
        return <FaHeart className="art-type-icon tulip" />;
      case 'swan':
        return <FaStar className="art-type-icon swan" />;
      case 'rosetta':
        return <FaStar className="art-type-icon rosetta" />;
      default:
        return <FaStar className="art-type-icon other" />;
    }
  };

  /**
   * Returns color based on confidence score
   * @param {number} score - Confidence percentage (0-100)
   * @returns {string} - Hex color code
   */
  const getScoreColor = (score) => {
    if (score >= 80) return '#28a745'; // Green for high confidence
    if (score >= 60) return '#ffc107'; // Yellow for good confidence
    return '#dc3545'; // Red for low confidence
  };



  return (
    <div className="results-container fade-in">
      {/* MAIN RESULT CARD - Shows detected pattern and confidence */}
      <div className="main-result-card">
        <div className="result-header">
          {/* Icon representing the detected latte art type */}
          <div className="result-icon">
            {getArtTypeIcon(art_type)}
          </div>
          {/* Pattern name and description */}
          <div className="result-title">
            <h2>{art_type.charAt(0).toUpperCase() + art_type.slice(1)} Pattern Detected</h2>
            <p className="result-subtitle">
              {art_type === 'heart' && "Classic symmetrical heart shape"}
              {art_type === 'tulip' && "Elegant tulip with multiple petals"}
              {art_type === 'swan' && "Elegant swan with curved neck"}
              {art_type === 'rosetta' && "Beautiful rosetta with flowing petals"}
              {art_type === 'other' && "Unique creative design"}
            </p>
          </div>
          {/* Confidence percentage badge */}
          <div className="confidence-badge">
            <span className="confidence-percentage">{(confidence * 100).toFixed(1)}%</span>
            <span className="confidence-label">Confidence</span>
          </div>
        </div>
        
        {/* ANIMATED CONFIDENCE BAR - Visual representation of AI confidence */}
        <div className="confidence-visual">
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ 
                width: `${confidence * 100}%`, // Bar width matches confidence percentage
                background: getScoreColor(confidence * 100) // Color changes based on confidence
              }}
            ></div>
          </div>
          {/* Confidence level indicators */}
          <div className="confidence-levels">
            <span className={confidence >= 0.8 ? 'active' : ''}>High</span>
            <span className={confidence >= 0.6 && confidence < 0.8 ? 'active' : ''}>Good</span>
            <span className={confidence >= 0.4 && confidence < 0.6 ? 'active' : ''}>Moderate</span>
            <span className={confidence < 0.4 ? 'active' : ''}>Low</span>
          </div>
        </div>
      </div>

      {/* Tips Section */}
      <div className="tips-section fade-in">
        <h3>💡 Tips for Better Latte Art</h3>
        <div className="tips-grid">
          <div className="tip-card">
            <div className="tip-icon">🥛</div>
            <h4>Milk Texture</h4>
            <p>Create smooth, velvety microfoam for better pattern definition</p>
          </div>
          <div className="tip-card">
            <div className="tip-icon">🎯</div>
            <h4>Pouring Technique</h4>
            <p>Pour slowly and steadily with controlled movements</p>
          </div>
          <div className="tip-card">
            <div className="tip-icon">🌡️</div>
            <h4>Temperature Control</h4>
            <p>Keep milk at 60-65°C for optimal texture and pattern formation</p>
          </div>
          <div className="tip-card">
            <div className="tip-icon">📐</div>
            <h4>Symmetry</h4>
            <p>Keep your cup at a slight angle for even distribution</p>
          </div>
          <div className="tip-card">
            <div className="tip-icon">☕</div>
            <h4>Espresso Base</h4>
            <p>Ensure your espresso has a rich crema layer for better contrast</p>
          </div>
          <div className="tip-card">
            <div className="tip-icon">🎨</div>
            <h4>Practice</h4>
            <p>Start with simple designs before attempting complex patterns</p>
          </div>
        </div>
      </div>

      {/* Achievement Section */}
      {confidence >= 0.8 && (
        <div className="achievement-section fade-in">
          <div className="achievement-content">
            <div className="achievement-icon">🏆</div>
            <div className="achievement-text">
              <h3>Excellent Work!</h3>
              <p>Your latte art shows great technique and is easily recognizable!</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
