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
- **Pynvml:** [BSD-3-Clause License](https://pypi.org/project/pynvml/) (`pynvml>=11.5.0`)
- **Pydantic:** [MIT License](https://github.com/pydantic/pydantic/blob/main/LICENSE) (`pydantic>=2.0.0`)
- **huggingface-hub:** [Apache-2.0 License](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) (`huggingface-hub>=0.20.0`)

---

## 2. External Model Assets

Model weights are external data assets downloaded at runtime to persistent storage volumes. They are not bundled into the Git repository or Docker image layers:

### FLUX.2 Klein 4B NVFP4
- **Provider:** Official Black Forest Labs ([black-forest-labs/FLUX.2-klein-4b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4))
- **Format:** NVFP4 (NVIDIA FP4 for Blackwell Tensor Cores)
- **License:** Governed by Black Forest Labs FLUX.2 Model License agreement.
- **Attribution Note:** This uses the official Black Forest Labs NVFP4 release (not Kizuna's separate Flux2-klein-Lite GPTQ INT4 project).

### Qwen3-4B FP4 Flux2 Text Encoder
- **Provider:** Comfy-Org / Alibaba Cloud Qwen Team ([Comfy-Org/flux2-assets](https://huggingface.co/Comfy-Org/flux2-assets))
- **Format:** FP4 Safetensors
- **License:** Governed by Qwen Model License agreement.

### Flux2 VAE
- **Provider:** Black Forest Labs ([black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev))
- **Format:** Float16 / Bfloat16 AutoencoderKL
- **License:** Governed by Black Forest Labs FLUX.2 License agreement.

### v7 DistilBERT Prompt Safety Classifier
- **Base Architecture:** DistilBERT (`distilbert-base-uncased`, Apache 2.0)
- **Safety Heads:** Custom continuous multi-width regression heads for child safety (~12yo appliance)
