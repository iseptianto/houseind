FROM python:3.9-slim
WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==1.5.3 \
    streamlit==1.24.0 \
 && pip install --timeout 300 --retries 10 -r requirements.txt

COPY app.py translations.py ./
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
