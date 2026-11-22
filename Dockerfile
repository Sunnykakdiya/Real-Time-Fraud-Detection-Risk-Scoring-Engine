FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ALLOW_FILE_STORE=true \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Install system build dependencies for LightGBM/XGBoost/SHAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, frontend, and tests
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY tests/ ./tests/

# Create runtime directories
RUN mkdir -p data/raw data/processed models

# Startup entrypoint: auto-generates sample data and trains initial model if models/ is empty
RUN echo '#!/bin/bash\n\
if [ ! -f models/model_bundle.joblib ]; then\n\
  echo "[*] Initializing model artifacts..."\n\
  python src/generate_sample_data.py --n-samples 25000\n\
  python src/preprocess.py\n\
  python src/train.py --quick\n\
fi\n\
echo "[*] Launching Uvicorn FastAPI server on port 8000..."\n\
exec uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
