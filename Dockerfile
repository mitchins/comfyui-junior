FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="ComfyUI Junior" \
      org.opencontainers.image.description="Opinionated image-generation appliance powered by FLUX.2 Klein and Blackwell NVFP4" \
      org.opencontainers.image.authors="Mitchell Currie <mitch@mitchellcurrie.com>" \
      org.opencontainers.image.source="https://github.com/mitchins/comfyui-junior" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    COMFY_DIR="/app/ComfyUI" \
    MODEL_DIR="/models" \
    DATA_DIR="/data"

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    git \
    tini \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python3.10 -m venv /opt/venv && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with Blackwell / CUDA 13.0 support (exact pinned cu130 wheel, no fail-open fallback)
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# Install comfy-kitchen and core ML dependencies (quoted version specifiers to prevent shell redirection)
RUN pip install --no-cache-dir \
    comfy-kitchen==0.2.31 \
    "transformers>=4.40.0" \
    "safetensors>=0.4.0" \
    "pynvml>=11.5.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.0.0" \
    "pillow>=10.0.0" \
    "huggingface-hub>=0.20.0" \
    "torchsde>=0.2.6" \
    "einops>=0.8.0" \
    "spandrel>=0.4.0" \
    "scipy>=1.11.0" \
    "timm>=1.0.0"

# Clone pinned upstream ComfyUI (zero custom nodes, internal only)
WORKDIR /app
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    cd /app/ComfyUI && \
    git checkout 7fe8a6138504f90ff7be82f3babf416da32876b1 && \
    pip install --no-cache-dir -r requirements.txt

# Copy and install ComfyUI Junior package
WORKDIR /app/comfyui-junior
COPY pyproject.toml README.md ./
COPY config/ /app/config/
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Create volume mount points
RUN mkdir -p /models /data

WORKDIR /app
EXPOSE 8000

ENTRYPOINT ["tini", "--", "comfyui-junior"]
