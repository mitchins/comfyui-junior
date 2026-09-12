# Third-Party Licenses and Attributions

ComfyUI Junior is licensed under the MIT License. The appliance image and runtime stack incorporate or interoperate with third-party software and external model assets, which are governed by their respective licenses:

---

## 1. Runtime & Core Dependencies

### ComfyUI
- **Repository:** [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **Commit:** `7fe8a6138504f90ff7be82f3babf416da32876b1`
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Note:** ComfyUI runs as an independent internal backend service inside the appliance container. ComfyUI source code is not bundled into or mixed with ComfyUI Junior's MIT Python package.

### PyTorch & NVIDIA CUDA Runtime
- **PyTorch:** [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE) (BSD-style, version `2.11.0+cu130`)
- **NVIDIA CUDA Toolkit & Driver Userspace:** [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)

### comfy-kitchen
- **Repository:** [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)
- **Package:** PyPI release `comfy-kitchen==0.2.31`
- **License:** Apache License 2.0 (Accelerated kernels including native `scaled_mm_nvfp4`)

### Alpine.js
- **Repository:** [alpinejs/alpine](https://github.com/alpinejs/alpine)
- **Version:** `3.14.8`
- **License:** MIT License (Locally vendored in `src/comfyui_junior/static/vendor/alpine.min.js`)

### Direct Python Dependencies
- **FastAPI:** [MIT License](https://github.com/fastapi/fastapi/blob/master/LICENSE) (`fastapi>=0.115.0`)
- **Uvicorn:** [BSD-3-Clause License](https://github.com/encode/uvicorn/blob/master/LICENSE.md) (`uvicorn[standard]>=0.30.0`)
- **Transformers:** [Apache-2.0 License](https://github.com/huggingface/transformers/blob/main/LICENSE) (`transformers>=4.40.0`)
- **Safetensors:** [Apache-2.0 License](https://github.com/huggingface/safetensors/blob/main/LICENSE) (`safetensors>=0.4.0`)
- **Pillow:** [HPND License](https://github.com/python-pillow/Pillow/blob/main/LICENSE) (`pillow>=10.0.0`)
- **Pydantic:** [MIT License](https://github.com/pydantic/pydantic/blob/main/LICENSE) (`pydantic>=2.0.0`)
- **huggingface-hub:** [Apache-2.0 License](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) (`huggingface-hub>=0.20.0`)
- **fastText (Python bindings):** [MIT License](https://github.com/facebookresearch/fastText/blob/main/LICENSE) (`fasttext==0.9.3`, pinned in `docker/constraints.txt`; used for the English-envelope language gate)

---

## 2. External Model Assets

Model weights are external data assets downloaded at runtime to persistent storage volumes. They are not bundled into the Git repository or Docker image layers. Every asset below is pinned in `config/models.json` with an immutable revision and/or SHA-256, verified at provisioning time.

### FLUX.2 Klein 4B NVFP4 (blackwell_nvfp4 stack)
- **Provider:** Official Black Forest Labs ([black-forest-labs/FLUX.2-klein-4b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4))
- **Format:** NVFP4 (NVIDIA FP4 for Blackwell Tensor Cores)
- **Pinned Revision:** `1db2b2f776c24b76f1122e5f69ab1949fc620068`
- **SHA-256:** `d8c5007b6a3bbbdfd38538bbcef5101a55dfde81894f58d2e3c8701cdef3542b`
- **License:** Governed by Black Forest Labs FLUX.2 Model License agreement.

### FLUX.2 Klein 4B FP8 (ampere_fp8 stack)
- **Provider:** Official Black Forest Labs ([black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8))
- **Format:** FP8 storage, bf16 compute on Ampere (no native FP8 GEMM below SM89)
- **Pinned Revision:** `5b4408e59397a4a37ccb46afe426d8ed86379441`
- **SHA-256:** `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6`
- **License:** Governed by Black Forest Labs FLUX.2 Model License agreement.

### Qwen3-4B FP4 Flux2 Text Encoder (blackwell_nvfp4 stack)
- **Provider:** Comfy-Org / Alibaba Cloud Qwen Team ([Comfy-Org/vae-text-encorder-for-flux-klein-4b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b))
- **Format:** FP4 Safetensors
- **Pinned Revision:** `c6ffb44d9b43b6a635fded0c5723173c4bcca257`
- **SHA-256:** `3eab03a77adb0ee5304a4e677c5d10ac22f9049c1d7c894adca4f8bb39206ca8`
- **License:** Governed by Qwen Model License agreement.

### Qwen3-4B FP8-Mixed Text Encoder (ampere_fp8 stack)
- **Provider:** Comfy-Org / Alibaba Cloud Qwen Team ([Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo), file `split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors`)
- **Format:** FP8-mixed Safetensors (identical Qwen3-4B base as the FP4 encoder)
- **Pinned Revision:** `08d04455279082882deaabc8d0d09fc914c071e1`
- **SHA-256:** `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15`
- **License:** Governed by Qwen Model License agreement (Apache-2.0 base model terms per Qwen release).

### Flux2 VAE (both stacks)
- **Provider:** Black Forest Labs ([black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev), file `ae.safetensors`)
- **Format:** Float16 / Bfloat16 AutoencoderKL
- **Pinned Revision:** `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`
- **SHA-256:** `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`
- **License:** Governed by Black Forest Labs FLUX.2 License agreement.

### v29db Prompt Safety Classifier
- **Distribution:** [mitchins/comfyui-junior-safety](https://huggingface.co/mitchins/comfyui-junior-safety)
- **Pinned Revision:** `a377d25017065db0ec5b8a7b6a2d11a30a906d5b` (immutable commit SHA; per-file SHA-256 digests in `config/models.json`, re-verified at classifier load)
- **Base Architecture:** DeBERTa-v3-base (12 layers, hidden 768; `microsoft/deberta-v3-base` lineage, Apache-2.0 / MIT per Microsoft release)
- **Safety Heads:** Six cumulative-logit ordinal heads (sexual, nudity, violence_gore, substances, disturbing, fetish) with policy_v6 binary PASS/BLOCK thresholds.
- **Serving:** FP16 on CUDA, single-view; see `docs/RTX3060_QUALIFICATION.md` for FP16 parity receipts.

### fastText lid.176 Language Identification
- **Distribution:** [facebookresearch/fastText release v0.9.2](https://github.com/facebookresearch/fastText/releases/download/v0.9.2/lid.176.bin)
- **SHA-256:** `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e`
- **License:** fastText is MIT-licensed; the lid.176 model is distributed under Creative Commons Attribution-Share-Alike 3.0 (CC-BY-SA 3.0).
- **Use:** English-envelope gate at a documented operating point (top-1 != en AND p >= 0.60 -> refusal).
