import React from 'react';
import { FaCoffee } from 'react-icons/fa';
import './Header.css';

/**
 * Header Component
 * Displays app branding and title
 */
const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        {/* LOGO SECTION - App icon and title */}
        <div className="logo">
          <div className="logo-icon-container">
            <FaCoffee className="logo-icon" />
          </div>
          <h1>Latte Art Classifier</h1>
        </div>
        {/* TAGLINE - Brief description */}
        <p className="tagline">AI-Powered Pattern Recognition</p>
      </div>
    </header>
  );
};

export default Header;
