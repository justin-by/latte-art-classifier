# Heroku Deployment Guide

## 🚀 Deploying Latte Art Classifier to Heroku

### Prerequisites
- Heroku CLI installed (`brew install heroku/brew/heroku` on Mac)
- Git repository set up
- Kaggle account with API key (optional - model is already trained)

### 1. Heroku App Setup

```bash
# Create Heroku app
heroku create your-latte-art-classifier

# Set buildpacks (Python + Node.js)
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/nodejs
```

### 2. Environment Variables (Security)

Set your Kaggle credentials as environment variables (optional since model is pre-trained):

```bash
# Set Kaggle API credentials (only needed if retraining)
heroku config:set KAGGLE_USERNAME=justinsungby
heroku config:set KAGGLE_KEY=your_kaggle_api_key_here

# Set Flask environment
heroku config:set FLASK_ENV=production
```

### 3. Required Files

#### `Procfile` (create in root directory)
```
web: cd backend && python app.py
```

#### `requirements.txt` (already exists in backend/)
```
flask==2.3.3
flask-cors==4.0.0
tensorflow==2.13.0
opencv-python==4.8.1.78
numpy==1.24.3
pillow==10.0.1
scikit-learn==1.3.0
werkzeug==2.3.7
python-dotenv==1.0.0
```

#### `package.json` (already exists in root)
```json
{
  "name": "latte-art-classifier",
  "version": "1.0.0",
  "scripts": {
    "start": "cd backend && python app.py",
    "build": "cd frontend && npm run build"
  }
}
```

### 4. Deployment Process

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit for Heroku deployment"

# Add Heroku remote
heroku git:remote -a your-latte-art-classifier

# Deploy to Heroku
git push heroku main
```

### 5. Post-Deployment

```bash
# Check logs to ensure everything is working
heroku logs --tail

# Open the app in browser
heroku open

# Check app status
heroku ps
```

**Note**: The model (`kaggle_latte_art_model.h5`) is already trained and included in the deployment, so no additional setup is needed!

### 6. Security Notes

- ✅ Kaggle API key is stored as environment variable (secure)
- ✅ No sensitive files in git repository
- ✅ `.gitignore` protects credentials
- ✅ Environment variables are encrypted on Heroku

### 7. Troubleshooting

#### Common Issues:

1. **Build timeout**: Increase build timeout in Heroku settings
2. **Memory issues**: Upgrade to a higher Heroku plan
3. **Model loading**: Ensure model files are properly included
4. **CORS issues**: Check Flask-CORS configuration

#### Useful Commands:

```bash
# Check app status
heroku ps

# View environment variables
heroku config

# Run commands on Heroku
heroku run python backend/test_model.py

# Scale the app
heroku ps:scale web=1
```

### 8. Production Considerations

- **Model Size**: Consider using smaller models for faster loading
- **Caching**: Implement Redis for model caching
- **CDN**: Use CloudFront for static assets
- **Monitoring**: Set up Heroku metrics and alerts

## 🔒 Security Best Practices

1. **Never commit API keys** to git
2. **Use environment variables** for all secrets
3. **Regularly rotate** API keys
4. **Monitor usage** and costs
5. **Use HTTPS** in production
