# Third-Party Licenses and Attributions

ComfyUI Junior is licensed under the MIT License. The appliance image and runtime stack incorporate or interoperate with third-party software and external model assets, which are governed by their respective licenses:

---

## 1. Runtime & Core Dependencies

### ComfyUI
- **Repository:** [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Note:** ComfyUI runs as an independent internal backend service inside the appliance container. ComfyUI source code is not bundled into or mixed with ComfyUI Junior's MIT Python package.

### PyTorch & NVIDIA CUDA Runtime
- **PyTorch:** [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE) (BSD-style)
- **NVIDIA CUDA Toolkit & Driver Userspace:** [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)

### comfy-kitchen
- **Repository:** [comfyanonymous/comfy-kitchen](https://github.com/comfyanonymous/comfy-kitchen)
- **License:** MIT License / Apache 2.0 (Accelerated kernels including `scaled_mm_nvfp4`)

### Alpine.js
- **Repository:** [alpinejs/alpine](https://github.com/alpinejs/alpine)
- **License:** MIT License (Locally vendored in `src/comfyui_junior/static/vendor/alpine.min.js`)

### FastAPI, Uvicorn, Transformers, Safetensors, Pillow, Pynvml
- **FastAPI / Uvicorn / Safetensors / Pynvml:** MIT / BSD Licenses
- **Transformers:** Apache License 2.0

---

## 2. External Model Assets

Model weights are external data assets downloaded at runtime to persistent storage volumes. They are not bundled into the Git repository or Docker image layers:

### FLUX.2 Klein 4B NVFP4
- **Provider:** Black Forest Labs / Kizuna
- **Format:** NVFP4 (NVIDIA FP4 for Blackwell Tensor Cores)
- **License:** Governed by Black Forest Labs FLUX.2 Model License agreement.

### Qwen3-4B FP4 Flux2 Text Encoder
- **Provider:** Comfy-Org / Alibaba Cloud Qwen Team
- **Format:** FP4 Safetensors
- **License:** Governed by Qwen Model License agreement.

### Flux2 VAE
- **Provider:** Black Forest Labs
- **Format:** Float16 / Bfloat16 AutoencoderKL
- **License:** Governed by Black Forest Labs FLUX.2 License agreement.

### v7 DistilBERT Prompt Safety Classifier
- **Base Architecture:** DistilBERT (`distilbert-base-uncased`, Apache 2.0)
- **Safety Heads:** Custom continuous regression heads for child safety (~12yo appliance)
