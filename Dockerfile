# Base image untuk dependensi Python umum
FROM python:3.9-slim as base
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# First install numpy to ensure consistent binary interface
RUN pip install --no-cache-dir numpy==1.24.3

# Then install other core numeric libraries and ML dependencies
RUN pip install --no-cache-dir \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    xgboost==1.7.3 \
    joblib==1.3.1

# Install MLflow separately to avoid dependency conflicts
RUN pip install --no-cache-dir mlflow==2.8.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage untuk FastAPI app
FROM base as fastapi
WORKDIR /app

COPY src/api/ ./src/api/
RUN touch src/__init__.py src/api/__init__.py

# Create necessary directories
RUN mkdir -p /app/models/trained

# MLflow setup terintegrasi
RUN mkdir -p /mlflow && \
    ln -s /mlflow/mlflow.db /app/mlflow.db && \
    ln -s /mlflow/artifacts /app/mlruns

ENV PYTHONPATH=/app
EXPOSE 8000

# Default command untuk FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
