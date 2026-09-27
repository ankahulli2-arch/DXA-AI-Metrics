FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

COPY entrypoint.sh /app/entrypoint.sh
COPY Final_sum.py /app/Final_sum.py
COPY Visual.py /app/Visual.py
COPY api.py /app/api.py
COPY config/ ./config/

RUN mkdir -p /app/out /app/visual_out && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]