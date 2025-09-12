# 🚀 Quick Start Guide

Get the Latte Art Classifier running in minutes!

## Prerequisites

Make sure you have the following installed:
- **Python 3.8+** (check with `python3 --version`)
- **Node.js 14+** (check with `node --version`)
- **npm** (comes with Node.js, check with `npm --version`)

## Option 1: One-Command Start (Recommended)

Simply run:
```bash
./start.sh
```

This script will:
- ✅ Check all prerequisites
- 📦 Install all dependencies automatically
- 🚀 Start both backend and frontend servers
- 🌐 Open the application in your browser

## Option 2: Manual Setup

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup (in a new terminal)
```bash
cd frontend
npm install
npm start
```

## Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001

## Test the API

Run the test script to verify everything is working:
```bash
cd backend
python test_api.py
```

## Usage

1. **Upload an image** of your latte art
2. **Click "Analyze Latte Art"**
3. **View your results** including:
   - Art type classification
   - Overall grade (A+ to F)
   - Detailed scores for symmetry, contrast, and sharpness
   - Personalized feedback

## Troubleshooting

### Port Already in Use
If you get port conflicts:
- Backend: Change port in `backend/app.py` (line with `app.run()`) - currently using port 5001
- Frontend: React will automatically suggest an alternative port

### Python Dependencies Issues
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

### Node.js Dependencies Issues
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Stop the Application

Press `Ctrl+C` in the terminal where you ran the start script.

---

**Happy brewing! ☕✨**