FROM python:3.12-slim AS app-base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NN2GUI_DATA=/data TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
WORKDIR /app
RUN useradd --create-home app && mkdir /data && chown app:app /data
COPY requirements-ui.txt ./
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-ui.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "src/proyecto/app.py", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false", "--server.maxUploadSize=20"]

# Las dependencias CUDA de ambos frameworks se aíslan para evitar conflictos.
FROM app-base AS cuda
USER root
COPY requirements-worker.txt /app/requirements-worker.txt
RUN python -m venv /opt/torch \
    && /opt/torch/bin/pip install --no-cache-dir -r requirements-worker.txt \
    && /opt/torch/bin/pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
RUN python -m venv /opt/tensorflow \
    && /opt/tensorflow/bin/pip install --no-cache-dir -r requirements-worker.txt 'tensorflow[and-cuda]==2.18.0'
USER app

FROM app-base AS cpu
