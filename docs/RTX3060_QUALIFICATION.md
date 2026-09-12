# RTX 3060 / Ampere Requalification — ComfyUI Junior

> Repository note (v2.0.0): this is the qualification record from the runtime
> workspace where the requalification was performed. Receipt files under
> `benchmark_results/`, contact sheets under `output/` and live-stack test
> scripts under `tests/` referenced below live in that workspace, not in this
> repository. The shipped evaluation-instance mechanism is consolidated into
> the single `JUNIOR_EVALUATION_INSTANCE` environment variable (see README);
> the separate `/internal/eval` endpoint described here was the POC-era
> implementation that the consolidated design replaces.

Migration record appended to the Blackwell qualification in LEDGER.md / JOURNAL.md
(which are preserved unchanged). This file is the new Ampere qualification log.

## 1. Baseline Host Inventory (repaired machine, before any changes)

| Item | Value |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3060 (12 GB — 12288 MiB nominal, 11.63 GiB usable) |
| Compute capability (detected, not assumed) | **(8, 6) — Ampere SM86** |
| Driver | 595.71.05 / CUDA 13.2 |
| CUDA runtime (torch) | 13.0 (`torch 2.11.0+cu130`) |
| Python | 3.10.19 (conda env `torch`) |
| PyTorch | 2.11.0+cu130, `cuda.is_available() == True` |
| Key packages | transformers 5.6.0, fasttext 0.9.3, safetensors 0.7.0, huggingface_hub 1.18.0, comfy-kitchen 0.2.31 (pip), comfy-aimdo 0.4.13 |
| ComfyUI checkout | `7fe8a613` (recent core; frontend pkg 1.49.6) |
| ComfyUI models present | `flux-2-klein-4b-fp8.safetensors` (4.07 GB), `flux-2-klein-4b-nvfp4.safetensors` (2.46 GB, **Blackwell-only**), `qwen_3_4b_fp4_flux2.safetensors` (3.85 GB, **Blackwell-only FP4 TE**), `flux2-vae.safetensors` (0.34 GB) |
| fastText lid.176 | present at `ComfyUI/models/safety/lid.176.bin` |
| Junior repo state | `safe_image_proxy` still runs the **old v7 DistilBERT** filter (`v7_distilbert/`), with a `SAFETY_ENABLED` env kill-switch in `config.py`/`app.py` — predates v29db/policy_v6; to be replaced |

### Stale Blackwell artifacts detected (inactive after this requalification)
- `flux-2-klein-4b-nvfp4.safetensors` + NVFP4 banked workflow `workflows/flux2_klein_4b.json` (workflow loads the NVFP4 UNET — invalid on SM86).
- `qwen_3_4b_fp4_flux2.safetensors` FP4 text encoder (requires SM100+ native FP4 GEMM path).
- `launch_comfy.py` hard 10.0 GiB allocator cap tuned for the old 16 GB card / 10 GB envelope.
- Old journal/ledger describe `scaled_mm_nvfp4` kernel dispatch — not applicable on SM86.

### Prior-task state (what "Gemini" left)
- No v29db classifier download, no FP16 serving path, no wellformed/language gates,
  no analysis mode, no bypass-isolation tests. Task effectively not started;
  only the historical v7 Blackwell stack was on disk.

### Compute-capability probe (explicit)
`torch.cuda.get_device_capability(0) == (8, 6)` → Ampere. FP8 tensor-core GEMM
(`_scaled_mm` fp8) is **not native** on SM86; Comfy runs fp8 checkpoints by
dequant-to-bf16/fp16 compute with fp8 storage. NVFP4 checkpoints are not executable.

No packages were changed prior to this report.

---

## 2. Prompt-safety classifier: v29db FP16 restoration

**Frozen artifact:** `Mitchins/comfyui-junior-safety` @ `a377d25017065db0ec5b8a7b6a2d11a30a906d5b`
(sha256-verified pinned copy in `junior_safety_v29db/` — encoder + heads hashes
match the hub revision exactly).

**Architecture verified at load:** DeBERTa-v3-base (12 layers, hidden 768,
DebertaV2Model), masked-mean pooling over `last_hidden_state`, six
cumulative-logit ordinal heads widths [3,2,2,2,2,2] over
[sexual, nudity, violence_gore, substances, disturbing, fetish], policy_v6
binary PASS/BLOCK with `BLOCK_AT = {sexual:2, nudity:2, violence_gore:2,
substances:2, disturbing:2, fetish:1}`. Checkpoint schema mismatch → refuse to serve.

**Serving profile (per v29 rationale, confirmed against the published model card):**
single-view raw text, NO production canonicalisation (the two-view canonicalizer
stays inside the frozen HF artifact for parity only), well-formedness +
English-envelope gates in front. The old v7 DistilBERT filter and its
`SAFETY_ENABLED` env kill-switch were deleted — production safety is
unconditional.

**FP16 qualification (tests/parity_v29db_fp16.py, 27-probe frozen set):**

| Metric | Value |
| :--- | :--- |
| `next(model.parameters()).dtype` | **torch.float16** (enforced at load; refusal to serve otherwise) |
| Parity vs FP32 reference (single view) | **PASS — identical ordinal levels and PASS/BLOCK decisions, 27/27** |
| Max per-threshold probability drift | 0.429 (confined to post-gap ordinal thresholds that cannot affect decoded levels) |
| Weight VRAM | **350.6 MiB** |
| Cold load | 0.74 s |
| Warm single-prompt latency | **16.6 ms** (median of 21) |

Reference vs production semantics note: the leet probe ("n4k3d 13 y34r 0ld
g1rl") blocks even single-view in v29db (sexual:3), and is additionally
rejected earlier by the well-formedness gate (W3) — defense in depth.

## 3. Input-envelope gates

* **wellformed.py** (stage 1): deterministic anti-obfuscation — zero-width
  chars (W1), mixed-script homoglyphs in Latin text (W2), interleaved leet
  digits with an edge-digit-block exemption so "4k/ps5/covid19/co2" pass while
  "n4k3d/s3x" fail (W3), spaced-out words ≥5 single letters (W4), structural
  junk (W5: empty/>600 chars/control chars/run>31). 13/13 gate cases pass,
  zero false positives on child-style prompts with numbers.
* **langgate.py** (stage 2): fastText lid.176 (`ComfyUI/models/safety/lid.176.bin`)
  at operating point **top-1 ≠ en AND p ≥ 0.60 → unsupported_language**,
  <3-word prompts pass through. fasttext 0.9.3's `.predict()` is NumPy-2.x
  incompatible; the native `model.f.predict` binding is used instead (same
  model/prediction, no environment change). 8/8 gate cases pass (clear
  en/fr/es/de/ru separated; short/uncertain text falls through to the
  classifier — English-only promise, no multilingual moderation claim).
* **pipeline.py** composes the three stages, fails closed on classifier error,
  and emits exactly the three public failure types:
  `prompt_format_invalid`, `unsupported_language`, `content_policy_violation`.

## 3b. Colouring-sheet style template (explicit, user-selected)

Finding (from the bake-off defect suite): colouring-sheet wording renders a
COLOURED illustration inconsistently (25–60% saturated pixels depending on
seed/exact phrasing; "colouring sheet" sometimes comes out empty, "colouring
in sheet" reliably coloured — model variance, not quantization). Product
intent for that use is an EMPTY printable outline sheet.

Design (revised per operator review — no transparent rewriting): the
behaviour is an **explicit template**. The request carries
`style: "none" | "colouring_sheet"` (strict server-side allowlist,
`invalid_style` → 400 on anything else); the selected template expands to a
fixed additive suffix BEFORE the safety gates (classify-what-you-render) and
is reported in `meta.style_template` / `revised_prompt`. The Imagine studio
exposes it as a "Picture / Colouring sheet" toggle next to aspect and quality
(persisted in local history). No implicit prompt sniffing exists anywhere.

Measured (fp8-storage Klein, 1024²):

| request | sat% | white% | result |
| :--- | ---: | ---: | :--- |
| bare prompt, style=none | varies by seed/wording | | coloured OR empty (unreliable) |
| style=colouring_sheet | **0.4** | **92.0** | empty, printable |

Safety unchanged: "a colouring sheet of a naked woman" + template still
BLOCKs (content_policy_violation); §14 suite qualified 22/22; bypass
isolation 18/18 after the change. Tests: `tests/test_styles.py`
(allowlist/expansion/no-sniffing/blank-prompt rejection).

## 4. Ampere image backend evaluation

Detected compute capability (8,6)/SM86 — no native FP8 GEMM, NVFP4 emulated
only (confirmed by comfy-kitchen "Native ops … emulated ops: nvfp4" line).
The old NVFP4 workflow/TE are invalid on this machine.

**Nunchaku SVDQuant INT4 (Z-Image)** — evaluated and rejected for this
appliance: PyPI ships only a pure-python wheel (no prebuilt cp310/torch2.11/
cu130 kernels; GitHub releases unreachable/unverifiable from this host ⇒
JIT/source CUDA build), plus a custom Comfy node loader — fails the
"reproducible pinned dependencies / no custom plugin ecosystem" requirement.
Revisit if a clean wheel matrix for our stack appears.

Both qualified candidates use ONE shared Ampere-compatible text encoder
(`qwen_3_4b_fp8_mixed.safetensors`, official Comfy-Org artifact) and zero
custom nodes.

### Candidate A — FLUX.2 Klein 4B FP8 (`flux-2-klein-4b-fp8.safetensors`)
Official BFL fp8 checkpoint, flux2 CLIP type, existing flux2 VAE, tiled VAE
decode (512/64), euler/simple, cfg 1.0. Steps: normal 4 / high 8.

### Candidate B — Z-Image Turbo INT8 ConvRot (`z_image_turbo_int8_convrot.safetensors`)
Official Comfy-Org int8_convrot checkpoint (torchao 0.17.0 already present),
lumina2 CLIP type, `ae.safetensors` VAE, ModelSamplingAuraFlow shift 3,
res_multistep/simple, cfg 1.0. Steps: normal 8 / high 16.

### §12 comparison (1024², warm medians; full data in benchmark_results/)

| | FLUX.2 Klein FP8 | Z-Image Turbo int8 |
| :--- | :--- | :--- |
| generation quality | subject to manual sign-off (contact sheet) | subject to manual sign-off (contact sheet) |
| prompt following | not visually verified here (no vision in requalification session) | same |
| text rendering | same — see SPACE CLUB row of contact sheet | same |
| 1024² latency (normal) | **9.2 s** | 11.8 s |
| 1024² latency (high) | **15.1 s** | 20.4 s |
| cold start (model load) | 26–30 s | 37–41 s |
| peak VRAM (NVML process) | **9630 MiB** | 9888 MiB |
| idle residency | 9.6 GiB (TE+UNET+VAE resident) | 9.9 GiB |
| CPU offload events | none (weights fully resident, zero streamed reloads warm) | none |
| dependency complexity | fp8 ckpt + shared fp8 TE + existing VAE | int8_convrot ckpt + shared TE + VAE (torchao) |
| Comfy custom nodes | **none** | **none** |
| reproducibility | official BFL/Comfy-Org artifacts, pinned filenames | official Comfy-Org artifacts, pinned filenames |
| suitability for 3060 12GB | **chosen** — faster at every quality, lower VRAM, stack continuity with the FLUX.2-based product | qualified fallback (swap `WORKFLOW_PATH_ZIMAGE`) |

**Decision: FLUX.2 Klein 4B FP8** as the RTX 3060 production backend
(`flux2-klein-fp8-safe`). Final visual quality sign-off pending human review of
`output/contact_sheet_1024sq_normal.png` (both backends, 6 prompts, labeled).
Text-rendering-heavy use (posters) should be checked especially.

## 5. Memory target

Measured steady state, single backend resident: ComfyUI **9622 MiB** + proxy
(FP16 classifier + gates) **~510 MiB** + desktop ~273 MiB ≈ **10.4 GiB of
11.63 GiB usable** — fits with ~1.2 GiB headroom, zero OOMs across 80+
generations, no weight streaming warm.

* The Blackwell **10 GiB hard allocator cap is NOT reused**. Measured behavior
  shows no need for a cap at qualified request shapes; a cap would only
  force eviction churn.
* Oversized-request guardrail lives at the **API boundary** instead: per-side
  size clamped to ≤1344 px (the frontend offers 768/1024/1344 only), so the
  measured working set cannot be blown up by request shape. Tiled VAE decode
  (512/64) retained everywhere.
* No OOM-exception control flow anywhere in the serving path.

## 6. Safety architecture (production vs analysis mode)

**Production** (`POST /v1/images/generations`): wellformed → language → FP16
v29db → policy_v6 → PASS → generation. Unconditional: no `SAFETY_ENABLED`
env, no request field, no header, no query parameter can weaken it. Classifier
load is a hard startup dependency (service refuses to start otherwise);
runtime classifier errors fail closed.

**Analysis mode** (`POST /internal/eval/render-and-classify`, module
`safe_image_proxy/analysis.py`): separate router, mounted once at import time
only when the process was started with `JUNIOR_ENABLE_IMAGE_ANALYSIS_MODE=1`
(default 0 → route does not exist; GET returns 404). No runtime toggle. The
module shares no request path with production and never imports the pipeline;
the pipeline never imports it. Render is unrestricted **on that endpoint
only**, goes through `PreviewImage` (ComfyUI temp, never the output store),
history entry is scrubbed (`POST /history {"delete":[…]}`), the temp file is
deleted, the image transits only to `/dev/shm/comfy-junior-eval/<uuid>.png`
and is unlinked in `finally` on success/VLM failure/classifier failure/
exception. Startup cleans stale eval files. The endpoint returns
classifications only — no bytes, URL, or base64 — and never sees the frontend.

**VLM judge:** OpenAI-compatible (`IMAGE_ANALYSIS_BASE_URL/_MODEL/_API_KEY`),
receives the image + fixed rubric only — **never the source prompt**. Returns
sexual/nudity/violence/gore/substances/disturbing/overall/confidence/evidence.
Optional local specialist classifier hook preserved via
`IMAGE_ANALYSIS_LOCAL_CLASSIFIER` (unset — nothing heavyweight bundled).
Verified end-to-end against `tests/mock_vlm.py` (which also asserts the VLM
saw 1 image and rubric-only text).

## 7. Test results (all hard acceptance criteria)

* FP16 parity: **PASS 27/27** (`tests/parity_v29db_fp16.py`)
* Gate unit checks: wellformed 13/13, langgate 8/8
* Bypass isolation, default config: **18/18** (`tests/test_bypass_isolation.py`)
* Bypass isolation, analysis mode: **23/23** (incl. render 200, VLM classify,
  no image data in response, /dev/shm + ComfyUI temp clean)
* Final qualification §14: **22/22 — QUALIFIED** (`tests/final_qualification.py`,
  result JSON in `tests/final_qualification_result.json`)

## 8. Environment / artifacts of record

* Backend: ComfyUI `7fe8a613` (core 0.33.0) native on 127.0.0.1:8189 via
  `launch_comfy.py` (allocator-cap code present but **unset** for this card;
  `COMFY_ALLOCATOR_CAP_GIB` available if ever needed).
* NOTE: a **stale Docker ComfyUI on port 8188** (old NVFP4 deployment, own
  model copy, 7.5 GiB cap, exposed 0.0.0.0) is still running from before the
  repair — remove it with docker access (requires root). The proxy targets
  8189 explicitly; the stale container cannot serve Ampere generation.
* Retired-but-retained on disk (inactive): `flux-2-klein-4b-nvfp4.safetensors`,
  `qwen_3_4b_fp4_flux2.safetensors`, `v7_distilbert/`, old NVFP4 workflow
  (`workflows/flux2_klein_4b.json` superseded by `flux2_klein_fp8_ampere.json`).
* Launch commands:
  * `python launch_comfy.py --listen 127.0.0.1 --port 8189 --disable-auto-launch`
  * `COMFY_BASE_URL=http://127.0.0.1:8189 python -m uvicorn safe_image_proxy.app:app --host 0.0.0.0 --port 8000`
  * analysis instance adds `JUNIOR_ENABLE_IMAGE_ANALYSIS_MODE=1` (+ VLM env),
    port 8010, bind 127.0.0.1 recommended.
* Frontend unchanged in design; now sends `quality`, no hardcoded model id,
  and renders all three friendly failure messages.
