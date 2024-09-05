# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set environment variables
ARG FLASK_ENV=production
ENV FLASK_ENV=${FLASK_ENV}
ENV FLASK_APP=app.py

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# If in development mode, install additional packages
RUN if [ "$FLASK_ENV" = "development" ] ; then pip install pytest flask-debugtoolbar ; fi

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /app/temp

# Expose port
EXPOSE 5000

# Run the application
CMD if [ "$FLASK_ENV" = "development" ] ; then \
        flask run --host=0.0.0.0 --port=5000 ; \
    else \
        gunicorn --bind 0.0.0.0:5000 app:app ; \
    fi