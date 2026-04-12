FROM python:3.9.6-slim

WORKDIR /app

# Install system dependencies for LightGBM/XGBoost
RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-fast.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --progress-bar off -r requirements.txt

# Copy application code FIRST
COPY . .

# Ensure models are explicitly included (important clarity step)
COPY models /app/models

# Set environment (optional but good practice)
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models

# Expose port
EXPOSE 5000

# Run production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "app:app"]