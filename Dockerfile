# Use a lightweight python image as base
FROM python:3.10-slim

# Install system dependencies needed for OpenCV, MediaPipe, and PyTorch (ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies and gunicorn for production
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application code
COPY . .

# Expose the port Flask runs on (Render overrides this if needed, but 5000 is default)
EXPOSE 5000

# Start the application using Gunicorn with Gevent WebSocket worker
CMD ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
