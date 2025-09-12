# Render Deployment Guide

This guide will help you deploy the Latte Art Classifier to Render with the full trained model.

## Prerequisites

1. A GitHub account
2. A Render account (free at [render.com](https://render.com))
3. Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

Make sure your repository contains:
- ✅ `backend/kaggle_latte_art_model.h5` (the trained model file)
- ✅ `Dockerfile`
- ✅ `render.yaml`
- ✅ `requirements.txt` in the root directory
- ✅ All source code

## Step 2: Deploy to Render

### Option A: Using Docker (Recommended)

1. **Go to Render Dashboard**
   - Visit [render.com](https://render.com) and sign in
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Connect your GitHub account if not already connected
   - Select your repository: `latte-art-classifier`

3. **Configure Service**
   - **Name**: `latte-art-classifier`
   - **Environment**: `Docker`
   - **Plan**: `Free` (should be sufficient for your model)
   - **Region**: Choose closest to your users
   - **Branch**: `main`

4. **Environment Variables** (if needed)
   - No additional environment variables required for basic deployment
   - The app will automatically detect the PORT environment variable

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy your application
   - This may take 5-10 minutes for the first deployment

### Option B: Using render.yaml

1. **Go to Render Dashboard**
   - Visit [render.com](https://render.com) and sign in
   - Click "New +" → "Blueprint"

2. **Connect Repository**
   - Connect your GitHub account if not already connected
   - Select your repository: `latte-art-classifier`

3. **Deploy**
   - Render will automatically detect the `render.yaml` file
   - Click "Apply" to deploy
   - This may take 5-10 minutes for the first deployment

## Step 3: Verify Deployment

1. **Check Build Logs**
   - Go to your service dashboard
   - Check the "Logs" tab for any build errors
   - Look for: "✅ Loaded trained Kaggle model"

2. **Test the Application**
   - Visit your Render URL (e.g., `https://latte-art-classifier.onrender.com`)
   - Upload a test image
   - Verify you get accurate predictions (not random ones)

3. **Test API Endpoints**
   ```bash
   # Health check
   curl https://your-app-name.onrender.com/api/health
   
   # Status check
   curl https://your-app-name.onrender.com/api/status
   ```

## Render Free Plan Limits

- **Build Time**: 90 minutes per month
- **Run Time**: 750 hours per month (enough for 24/7 operation)
- **Memory**: 512 MB RAM
- **Storage**: 1 GB
- **Bandwidth**: 100 GB per month

Your application should easily fit within these limits:
- Model file: ~24 MB
- Total app size: ~100-200 MB
- Memory usage: ~200-300 MB

## Troubleshooting

### Build Fails
- Check that `backend/kaggle_latte_art_model.h5` exists in your repository
- Verify all dependencies are in `requirements.txt`
- Check build logs for specific error messages

### Model Not Loading
- Ensure the model file is committed to your repository
- Check that the file path in the code matches the actual file location
- Look for "✅ Loaded trained Kaggle model" in the logs

### App Crashes
- Check the service logs for error messages
- Verify all environment variables are set correctly
- Ensure the PORT environment variable is being used

### Slow Performance
- Render free tier may have slower cold starts
- Consider upgrading to a paid plan for better performance
- The model loading happens on first request (lazy loading)

## Updating Your Deployment

1. **Push Changes to GitHub**
   ```bash
   git add .
   git commit -m "Update application"
   git push origin main
   ```

2. **Render Auto-Deploy**
   - Render will automatically detect changes
   - A new deployment will start automatically
   - Check the "Deploys" tab to monitor progress

## Cost

- **Free Plan**: $0/month
- **Starter Plan**: $7/month (if you need more resources)
- **Standard Plan**: $25/month (for production use)

The free plan should be sufficient for your latte art classifier!

## Support

- Render Documentation: [render.com/docs](https://render.com/docs)
- Render Community: [community.render.com](https://community.render.com)
- GitHub Issues: Create an issue in your repository for app-specific problems
