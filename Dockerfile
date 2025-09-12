# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json files and install Node dependencies
COPY package*.json ./
COPY frontend/package*.json ./frontend/
RUN npm install
RUN cd frontend && npm install

# Copy the entire project
COPY . .

# Build the React frontend
RUN cd frontend && npm run build

# Create uploads directory
RUN mkdir -p uploads

# Expose port (Render will set PORT environment variable)
EXPOSE 10000

# Start the Flask application
CMD ["python", "backend/app.py"]
