# 🚀 Quick Heroku Deployment Steps

## Step-by-Step Guide to Deploy Your Latte Art Classifier

### 1. Install Heroku CLI
```bash
# Mac
brew install heroku/brew/heroku

# Windows
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

### 2. Login to Heroku
```bash
heroku login
```

### 3. Create Heroku App
```bash
# Replace 'your-app-name' with your desired app name
heroku create your-latte-art-classifier
```

### 4. Set Buildpacks
```bash
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/nodejs
```

### 5. Initialize Git (if not already done)
```bash
git init
git add .
git commit -m "Initial commit for Heroku deployment"
```

### 6. Add Heroku Remote
```bash
heroku git:remote -a your-latte-art-classifier
```

### 7. Deploy to Heroku
```bash
git push heroku main
```

### 8. Open Your App
```bash
heroku open
```

## ✅ That's It!

Your app should now be live on Heroku! The trained model is already included, so no additional setup is needed.

## 🔧 Troubleshooting

If you encounter issues:

```bash
# Check logs
heroku logs --tail

# Check app status
heroku ps

# Restart the app
heroku restart
```

## 📝 Important Notes

- **Model Size**: The trained model (`kaggle_latte_art_model.h5`) is ~23MB and will be included in deployment
- **First Request**: The model loads lazily on the first request (may take 2-3 seconds)
- **Free Tier**: Heroku's free tier has limitations - consider upgrading for production use
- **Custom Domain**: You can add a custom domain in Heroku settings

## 🎯 Your App URL

After deployment, your app will be available at:
`https://your-latte-art-classifier.herokuapp.com`
