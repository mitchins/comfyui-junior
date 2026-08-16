# Qualified Hardware & Runtime Stack

## Reference Hardware Baseline
- **GPU:** NVIDIA GeForce RTX 5060 Ti 16 GB
- **Architecture:** Blackwell SM120 (Compute Capability 12.0)
- **NVIDIA Driver:** >= `595.71.05`
- **CUDA Runtime:** `13.0` / `12.8`

---

## Qualified Model Topology & Memory Layout

| Component | Asset Filename | Format / Quantization | Resident VRAM |
| :--- | :--- | :--- | :--- |
| **Diffusion Transformer** | `flux-2-klein-4b-nvfp4.safetensors` | NVFP4 (NVIDIA Native FP4) | **2.29 GB** |
| **Text Encoder** | `qwen_3_4b_fp4_flux2.safetensors` | FP4 Safetensors | **3.58 GB** |
| **VAE** | `flux2-vae.safetensors` | AutoencoderKL (Tiled 512×512, overlap 64) | **0.16 GB** |
| **Safety Classifier** | `v7_distilbert` | DistilBERT Base + 6 Regression Heads | **0.25 GB** |
| **Total Resident Models** | — | — | **~6.28 GB** |

---

## Native NVFP4 Kernel Acceleration
- **Kernel Dispatch:** 80 native `scaled_mm_nvfp4` hardware tensor core kernel dispatches per denoise step.
- **Dequantization / Upcasting:** None. Math executed directly in FP4 using Blackwell native tensor core instructions with per-tensor/per-row scale factors.

---

## Execution Configuration & Caching Allocator Fence
- **Denoise Schedule:** 4 steps, Euler sampler, simple schedule.
- **VAE Decode:** Explicit `VAEDecodeTiled(tile_size=512, overlap=64)`. Eliminates the ~11.5 GB transient tensor allocation of untiled VAE decode.
- **PyTorch Allocator Limit:** Hard ceiling of **10.0 GiB** enforced via `torch.cuda.memory.set_per_process_memory_fraction()`.
- **Observed Peak Allocator VRAM:** **6.67 GB** at 1024×1024.
- **Observed Process Peak VRAM:** **~7.45 GB** across all appliance processes.
- **Observed Warm E2E Latency:** **~3.25 s - 3.56 s** (Denoise ~1.39s, Tiled VAE ~1.43s, Safety ~4.1ms).
- **OOM Warnings / Fallbacks:** **0 (Zero)**.
- **Image Fidelity:** No observed fidelity regression compared to unconstrained 16 GB baseline.
