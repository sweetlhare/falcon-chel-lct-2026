FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=1000:1000 . .
RUN useradd --uid 1000 --user-group --create-home falcon && mkdir -p /storage /results && chown 1000:1000 /storage /results
USER 1000:1000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
