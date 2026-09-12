# ComfyUI Junior

A lightweight, high-speed, child-safe local image generation appliance powered by FLUX.2 Klein 4B, with an OpenAI-compatible API and a responsive browser frontend for phones, tablets and desktops.

## Architecture

```text
Browser (Phone, iPad, Desktop) / OpenWebUI
                 |
                 v  GET / (Imagine Studio) OR POST /v1/images/generations
+-----------------------------------------------------------+
|  Junior Appliance (:8000)                                 |
|                                                           |
|  * Responsive Imagine Frontend (Alpine.js + IndexedDB)    |
|  * OpenAI-Compatible Image API (/v1/images/generations)   |
|  * Prompt Safety Pipeline (unconditional):                |
|      well-formedness gate   -> prompt_format_invalid      |
|      English-envelope gate  -> unsupported_language       |
|      FP16 v29db classifier  -> content_policy_violation   |
|      policy_v6              -> PASS only, then generate   |
+----------------------------+------------------------------+
                             | (127.0.0.1:8188 internal only)
                             v
+-----------------------------------------------------------+
|  Internal ComfyUI Backend (pinned commit, no plugins)     |
|  * Stack: blackwell_nvfp4 (SM120) or ampere_fp8 (SM86+)   |
|  * Explicit VAEDecodeTiled (512x512, overlap 64)          |
+-----------------------------------------------------------+
```

## Hardware Requirements

Two qualified hardware stacks are supported, selected automatically from the detected GPU (or explicitly via `JUNIOR_IMAGE_STACK`):

| Stack | GPU requirement | Example | Warm 1024x1024 (normal) |
| --- | --- | --- | --- |
| `blackwell_nvfp4` | NVIDIA Blackwell SM120 (CC 12.0) | RTX 5060 Ti 16 GB | ~3.3 s |
| `ampere_fp8` | NVIDIA Ampere SM86 or newer (CC >= 8.6) | RTX 3060 12 GB | ~9.2 s |

Host prerequisites: NVIDIA display driver >= 595.71.05 and (for the Docker path) the NVIDIA Container Toolkit.

## Quick Start (Docker)

```bash
docker compose up -d
```

Then open http://localhost:8000/ .

### Volume Mounts and Model Acquisition

- `./models`: models are stored persistently on the host and verified idempotently on startup (immutable revisions and SHA-256 digests pinned in `config/models.json`).
- `./data`: ComfyUI runtime and temporary output directory.
- Zero models in Docker layers: the image builds with no model downloads. Missing public models for the selected stack are fetched and digest-verified at runtime.

## Features and Interfaces

### Imagine Web Studio (GET /)

- Clean and responsive: 48 px touch targets, iOS safe-area handling, Safari toolbar padding, zero external CDN dependencies (vendored Alpine.js).
- Local IndexedDB history of the latest 20 pictures.
- Portrait / square / landscape, normal / high quality, and explicit style templates: Picture, Real photo, Cartoon, Colouring sheet.
- Restrained child-friendly error states for the three public failure codes (scrambled text, English-only, content policy) without leaking backend internals.

### OpenAI-Compatible API (POST /v1/images/generations)

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute penguin astronaut looking at Earth",
    "model": "flux2-klein-safe",
    "size": "1024x1024",
    "style": "none",
    "quality": "normal",
    "response_format": "b64_json"
  }' | jq '.data[0].b64_json' -r | base64 -d > penguin.png
```

Request fields: `prompt` (1-1500 characters; prompts that cannot be fully classified after style expansion - including over-length ones - are rejected fail-closed with `prompt_format_invalid`), `size` (each side a multiple of 16 between 16 and 1344), `quality` (`normal`|`high`), `style` (`none`|`real_photo`|`cartoon`|`colouring_sheet`; explicit allowlist, reported back as `revised_prompt` and `meta.style_template`).

### Prompt Safety Pipeline (unconditional)

Three deterministic stages run on every appliance-mode request and no flag, parameter, header or environment toggle can weaken them:

1. **Well-formedness gate** - rejects obfuscated input (leet, homoglyphs, spaced-out words, zero-width characters) with `prompt_format_invalid`.
2. **English-envelope gate** - fastText `lid.176` refuses confidently non-English input with `unsupported_language`. The product promises English only and makes no multilingual-moderation claim.
3. **v29db classifier** - DeBERTa-v3-base with six cumulative-logit ordinal heads (sexual, nudity, violence_gore, substances, disturbing, fetish), served in FP16 on CUDA, deciding binary PASS/BLOCK under policy_v6. Artifact pinned to an immutable revision and re-verified against manifest digests at load; failures fail closed with zero ComfyUI submissions.

## Owner-Operated Evaluation Instances

For external measurement of the raw model (for example, gauging safety false positives/negatives with your own judge), an owner may deliberately launch an unfiltered evaluation instance. This is an infrastructure-level decision made at process start:

```bash
JUNIOR_EVALUATION_INSTANCE=1 docker compose ... # separate project, loopback binding recommended
```

Effects: prompt filtering is disabled by design, the child frontend is replaced by an API-only notice, startup logs a loud banner, and `/health` reports `"role": "evaluation"`. The request-handler code path is identical to appliance mode; only the injected gate differs. Never expose such an instance as the children's appliance.

## Configuration Reference

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Public service binding |
| `JUNIOR_IMAGE_STACK` | `auto` | `blackwell_nvfp4`, `ampere_fp8`, or auto-detect from the GPU |
| `COMFY_MEMORY_CAP_GIB` | per-stack | PyTorch allocator ceiling; 0 disables. Defaults: 10.0 (nvfp4), none (ampere_fp8) |
| `MODEL_DIR` / `DATA_DIR` | `/models` / `/data` | Persistent models and runtime data volumes |
| `SAFETY_DEVICE` | `cuda:0` | Device for the FP16 safety classifier |
| `SAFETY_MODEL_PATH` | `/models/safety/v29db` | Pinned v29db artifact directory (digest-verified) |
| `LID176_PATH` | `/models/safety/lid.176.bin` | fastText language-identification model |
| `JUNIOR_EVALUATION_INSTANCE` | `0` | Owner-operated unfiltered measurement mode (see above) |
| `HF_TOKEN` | empty | Optional Hugging Face token |

## Technical Details

Qualification records with benchmark receipts, memory measurements and safety-test results:

- `docs/QUALIFIED_STACK.md` - Blackwell SM120 / NVFP4 qualification (historical, preserved)
- `docs/RTX3060_QUALIFICATION.md` - RTX 3060 12 GB / Ampere fp8 requalification (v29db FP16 parity, gates, appliance behaviour)

## License and Attributions

ComfyUI Junior is licensed under the MIT License. Third-party dependencies and model assets retain their own licenses as documented in [THIRD_PARTY.md](THIRD_PARTY.md).
