# Base image
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TORCH_HOME=/app/models \
    ULTRALYTICS_CONFIG_DIR=/app/config

# Working directory
WORKDIR /app

# System dependencies for OpenCV / YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch 2.5.1 (compatible with YOLO weights)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose port
EXPOSE 10000

# Command to run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]