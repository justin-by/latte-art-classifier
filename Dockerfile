# Use Python 3.11 explicitly
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (including pre-built React frontend)
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=backend/app.py
ENV PYTHONPATH=/app

# Start the application
CMD ["python", "backend/app.py"]