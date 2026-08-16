# ComfyUI Junior ✨

Opinionated, high-speed, child-safe local image generation appliance powered by **FLUX.2 Klein 4B NVFP4** and NVIDIA Blackwell SM120.

```
Browser (Phone, iPad, Desktop) / OpenWebUI
                   │
                   ▼  GET / (Imagine Studio) OR POST /v1/images/generations
┌────────────────────────────────────────────────────────┐
│ Junior Appliance Container (:8000)                     │
│                                                        │
│ • Responsive Imagine Frontend (Alpine.js + IndexedDB)  │
│ • OpenAI-Compatible Image API (/v1/images/generations) │
│ • Inline v7 DistilBERT Safety Filter (~4.1 ms)         │
│   ├── BLOCK/ROUTE ──► 400 Refusal (0 Comfy jobs)       │
│   └── PASS        ──► Submit to Internal ComfyUI       │
└───────────────────────────┬────────────────────────────┘
                            │ (127.0.0.1:8188 internal only)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Internal ComfyUI Backend (fenced to 10.0 GiB cap)      │
│ • FLUX.2 Klein 4B NVFP4 (80 scaled_mm_nvfp4/step)      │
│ • Qwen3-4B FP4 Flux2 Text Encoder                      │
│ • Explicit VAEDecodeTiled (512x512, overlap 64)        │
│ • ~3.25 s warm 1024² generation (6.67 GB peak alloc)   │
└────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

- **GPU:** NVIDIA Blackwell GPU (Compute Capability 12.0 / SM120, e.g. **RTX 5060 Ti 16 GB**)
- **Host Drivers:** NVIDIA Display Driver >= `595.71.05`, NVIDIA Container Toolkit
- **Docker:** Docker Engine with GPU support

> [!NOTE]
> v1 targets the Blackwell SM120 native NVFP4 fast-path. Execution on older GPU architectures is not supported in this release.

---

## Quick Start (Docker)

Run the appliance with Docker Compose:

```bash
docker compose up -d
```

Open your browser:
👉 **`http://localhost:8000/`**

### Volume Mounts & Model Acquisition
- `./models`: Models are stored persistently on the host and verified idempotently on startup.
- `./data`: ComfyUI runtime and temporary output directory.
- **Zero models in Docker layers:** The Docker image builds with 0 model downloads. Missing public models are fetched automatically at runtime.

---

## Features & Interfaces

### 1. Imagine Web Studio (`GET /`)
- **Clean & Responsive:** Optimized for iPhone, iPad, and desktop with 48px touch targets, iOS safe-area handling, and Safari toolbar padding.
- **Client IndexedDB:** Caches the latest 20 generated pictures locally in your browser.
- **Zero External CDN Dependencies:** Bundles vendored Alpine.js for offline LAN operation.
- **Restrained Child-Friendly Error States:** Returns friendly guidance (*"That idea isn't available here. Try changing the picture a little."*) without leaking raw backend errors.

### 2. OpenAI-Compatible API (`POST /v1/images/generations`)
Connect OpenWebUI, scripts, or any OpenAI client:
- **Base URL:** `http://<HOST_IP>:8000/v1`
- **Model:** `flux2-klein-4b-safe`

Example `curl` request:
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute penguin astronaut looking at Earth",
    "model": "flux2-klein-4b-safe",
    "size": "1024x1024",
    "response_format": "b64_json"
  }' | jq '.data[0].b64_json' -r | base64 -d > penguin.png
```

### 3. Inline Semantic Safety Boundary
- **Architecture:** `DistilBertModel` with 6 continuous regression heads (`sexual`, `nudity`, `violence_gore`, `substances`, `disturbing`, `fetish`).
- **Performance:** **~4.1 ms** latency, **254 MB VRAM**.
- **Zero Submissions on Rejection:** Prompts triggering `BLOCK` or `ROUTE` policies return HTTP 400 immediately and submit 0 jobs to ComfyUI.

---

## Configuration Reference

Configure via `.env` or container environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Public interface binding |
| `PORT` | `8000` | Public service port |
| `COMFY_MEMORY_CAP_GIB` | `10.0` | PyTorch caching allocator ceiling |
| `MODEL_DIR` | `/models` | Path to persistent models volume |
| `DATA_DIR` | `/data` | Path to temporary output volume |
| `SAFETY_ENABLED` | `1` | Enable inline v7 DistilBERT safety filter |
| `SAFETY_DEVICE` | `cuda:0` | PyTorch device for safety classifier |
| `SAFETY_MODEL_PATH` | `/models/safety/v7_distilbert` | Local path to safety classifier weights |
| `HF_TOKEN` | *empty* | Optional Hugging Face token |

---

## Technical Details

For detailed benchmark logs, memory fences, and NVFP4 kernel execution receipts, see [`docs/QUALIFIED_STACK.md`](docs/QUALIFIED_STACK.md).

## License & Attributions

ComfyUI Junior is licensed under the [MIT License](LICENSE). Third-party dependencies (ComfyUI GPL-3.0, PyTorch, comfy-kitchen, model weights) retain their own licenses as documented in [`THIRD_PARTY.md`](THIRD_PARTY.md).
