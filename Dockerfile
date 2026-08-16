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
RUN python3.10 -m venv /opt/venv

# Copy and install complete locked dependency graph (including torch cu130, ComfyUI, and Junior deps)
WORKDIR /app
COPY docker/constraints.txt /app/constraints.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r /app/constraints.txt

# Clone pinned upstream ComfyUI (zero custom nodes, internal only)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    cd /app/ComfyUI && \
    git checkout 7fe8a6138504f90ff7be82f3babf416da32876b1

# Copy and install ComfyUI Junior package
WORKDIR /app/comfyui-junior
COPY pyproject.toml README.md ./
COPY config/ /app/config/
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Create volume mount points and unprivileged runtime user
RUN groupadd -g 1000 junior && \
    useradd -u 1000 -g junior -m -s /bin/bash junior && \
    mkdir -p /models /data /app && \
    chown -R junior:junior /app /models /data /opt/venv

WORKDIR /app
USER junior
EXPOSE 8000

ENTRYPOINT ["tini", "--", "comfyui-junior"]
