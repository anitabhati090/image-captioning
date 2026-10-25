# ---------- Stage 1: build dependencies ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch keeps the image small. Remove these two lines for a GPU build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader -d /opt/nltk_data wordnet omw-1.4

# ---------- Stage 2: lean runtime image ----------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA=/opt/nltk_data \
    HF_HOME=/home/appuser/.cache/huggingface
RUN useradd -m appuser
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-deps .

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
