# Klein-on-Ampere Bake-off: FP8 (incumbent) vs Unsloth GGUF Q4_K_M / Q5_K_M

> Repository note (v2.0.0): this is the qualification record from the runtime
> workspace where the requalification was performed. Receipt files under
> `benchmark_results/`, contact sheets under `output/` and live-stack test
> scripts under `tests/` referenced below live in that workspace, not in this
> repository. The shipped evaluation-instance mechanism is consolidated into
> the single `JUNIOR_EVALUATION_INSTANCE` environment variable (see README);
> the separate `/internal/eval` endpoint described here was the POC-era
> implementation that the consolidated design replaces.

Date: 2026-09-12 · Experimental only — **production profile untouched** (v29
safety stack and serving config unchanged throughout).

## Framing correction

The task text described the incumbent as "INT8/ConvRot". The actual qualified
Klein baseline on this machine is **fp8-storage → bf16-compute**
(`flux-2-klein-4b-fp8.safetensors`; SM86 has no FP8 tensor-core GEMM — fp8 is
storage only, math runs in bf16. Ampere's native integer path is INT8;
int8_convrot is the Z-Image checkpoint's quantization). Naming corrected
throughout; the bake-off compares the true incumbent against the GGUF
candidates.

## Defect-oriented quality suite (post-latency follow-up)

14 quant-sensitive prompts × 2 extra seeds on the colouring anecdotes, fixed
seeds, 1024², three arms (48 images). Objective signals — OWLv2 zero-shot
counting, Sobel edge density/components (linework busy-ness), mirror symmetry,
CLIP contrastive defect probes (extra legs / deformed hands / garbled text):

| Signal | FP8 | Q4_K_M | Q5_K_M |
| :--- | --- | --- | --- |
| penguin colouring count (both seeds) | **1 / 1** ✓ | **1 / 1** ✓ | **1 / 1** ✓ |
| edge components (busy-ness proxy) | 611 / 649 | 575 / 540 | 522 / 688 |
| six strawberries count | **6** ✓ | **6** ✓ | **6** ✓ |
| two cats on sofa | 2 ✓ | 2 ✓ | 2 ✓ |
| three sailboats | 0 ✗ | 1 ✗ | 1 ✗ |
| squirrel count | 1 ✓ | 1 ✓ | **2** ✗? |
| anatomy CLIP margins (horse/piano/bicycle/doctor/dog) | all positive | all positive | all positive |
| typography CLIP margin | ~16.5 | ~16.3 | ~16.4 (insensitive — needs eyes) |
| butterfly mirror deviation | 11.5 | 10.5 | 12.2 |

* The reported "multiple penguins on INT8" anecdote **did not reproduce** on
  any arm at these seeds (all single penguin; busy-ness components in the same
  band, Q4 marginally less busy).
* Counting holds to **6/6 strawberries** on all arms — quantization did not
  break numeracy at these levels.
* Sailboats-at-3 fails on all arms (either model or detector limit) — not a
  discriminator. One isolated Q5 anomaly (squirrel double-detection) is
  single-image anecdote-grade.
* **No reproducible measurable quality separation between the three arms.**
  Decisive artifact for human judgment:
  `output/contact_sheet_defect_suite.png` (16 rows × 3 arms, labelled).
  Typography ("HELLO SUN") especially needs human review — CLIP probes are
  insensitive to spelling.

## Baseline freeze (immutable comparator)

* Transformer: `flux-2-klein-4b-fp8.safetensors`, 4,070,624,520 B,
  sha256 `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6`
* TE: `qwen_3_4b_fp8_mixed.safetensors` (Comfy-Org/z_image_turbo split_files) —
  used identically in ALL arms (transformer-only A/B per protocol)
* VAE: `flux2-vae.safetensors`, `VAEDecodeTiled 512/64`
* Comfy `7fe8a6138504f90ff7be82f3babf416da32876b1`, torch 2.11.0+cu130,
  driver 595.71.05, comfy-kitchen 0.2.31, no allocator cap, API clamp ≤1344px
* Sampling: euler/simple, cfg 1.0, 4 steps (normal), fixed seeds 1000+i
* Stage decomposition: encode via Comfy node-cache delta (fresh vs cached text,
  same seed); per-step denoise via (12-step − 4-step)/8; VAE+overhead as
  remainder. VRAM: NVML process sampling at 10 Hz (pid-matched to the Comfy
  process; the initial Q4 VRAM pass mis-tracked the proxy and was re-run —
  see `benchmark_results/bakeoff/vram_pass.json`).

## GGUF arm setup (isolated experiment)

* Custom node: `city96/ComfyUI-GGUF` @ git `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
  (only custom node; no Manager)
* `gguf==0.17.1` (pip-pinned; node requires ≥0.13.0)
* Weights: `unsloth/FLUX.2-klein-4B-GGUF` @ HF rev `0084d1df98e2e2137fe776d55170bc4792ec1d66`
  * `flux-2-klein-4b-Q4_K_M.gguf` 2,604,311,104 B,
    sha256 `0b25d143c8469b342bc5af3bce92b783bf6b0636d285f7b2f75e38af63af9a15` ✓
  * `flux-2-klein-4b-Q5_K_M.gguf` 3,073,368,640 B,
    sha256 `58c01c75fee2272eadb127f53e26dd05a5a8e7f812e37c36fa44603301f91e54` ✓
* Same TE/VAE/sampler/seeds/resolutions as baseline; only node 1 changed
  (`UNETLoader` → `UnetLoaderGGUF`).

## §8 Decision table (warm, 1024² unless noted; 768² in parens where useful)

| Metric | FP8 (incumbent) | GGUF Q4_K_M | GGUF Q5_K_M |
| :--- | ---: | ---: | ---: |
| transformer size | 4.07 GB | 2.60 GB | 3.07 GB |
| idle VRAM (resident) | 9622 MiB | 8286 MiB | 8550 MiB |
| peak VRAM 1024² | **10702 MiB** | 9604 MiB | 9546 MiB |
| peak VRAM 768² | 10604 MiB | ~9.3–9.6 GiB | 9694 MiB |
| cold start (unload→image) | 26.3–29.6 s | 23.3 s | 20.1–23.3 s |
| text encode (warm) | ~0.06 s | ~0.09 s | ~0.10 s |
| denoise 4-step 768² | **3.01 s** | 4.12 s (+37%) | 4.45 s (+48%) |
| denoise 4-step 1024² | **5.54 s** | 6.57 s (+19%) | 6.60 s (+19%) |
| VAE (tiled) + overhead | ~2.5–3.7 s | ~2.5–3.7 s | ~2.4–4.5 s |
| warm E2E 768² | **5.56 s** | 6.59 s (+18%) | 6.88 s (+24%) |
| warm E2E 1024² | **9.19 s** | 10.22 s (+11%) | 11.10 s (+21%) |
| CPU offload / reloads (warm) | none | none | none |
| OOM events | 0 | 0 | 0 |
| CLIPScore 768/1024/portrait | 95.2 / 94.1 / 94.4 | 95.0 / 94.1 / 95.6 | 95.3 / 93.2 / 94.8 |
| same-seed deviation vs FP8 (MSE×100) | — | 1.28–1.82 | 0.81–0.96 |
| dependency burden | core | +1 custom node +`gguf` | +1 custom node +`gguf` |

(CLIPScore: ViT-B/32, 2.5·cos; same fixed prompts/seeds; the three arms are
statistically indistinguishable on prompt following. Deviation shows Q5 tracks
the FP8 reference more closely than Q4, as expected from the bitrate.)

## §7 Interpretation — what GGUF actually bought here

GGUF is a memory representation, not an Ampere speedup — confirmed: Q4/Q5
denoise is **19–48% slower** (runtime dequant/custom-op cost) while the whole
FP8 stack was already fully resident with zero offload. The VRAM "win" is
real but small in system terms — **~1.1 GiB peak / ~1.3 GiB idle (Q4)** —
because the shared 5.6 GB fp8-mixed Qwen encoder dominates residency in every
arm. There is no offload cliff for GGUF to avoid on this card at these sizes.

## §4 Text-encoder GGUF phase — deliberately not pursued

The TE swap's only benefit would be further residency reduction; residency is
not a constraint (FP8 fits with ~1.2 GiB headroom), and the Qwen/Klein GGUF
TE path is documented-risky. Per protocol ("do not patch around it; judge the
DiT alone"), the known-good `qwen_3_4b_fp8_mixed` encoder was kept in all
arms and the TE experiment is recorded as **not needed for the decision**.

## §9 Verdict

> ## **KEEP_CONVROT** — retain the fp8-storage incumbent (`flux-2-klein-4b-fp8.safetensors`)

Reason: the selection rule requires GGUF to buy something meaningful. It
bought ~1.1 GiB of unused headroom at an **11–24% E2E latency cost** and a
larger dependency surface (custom node + `gguf` pin), with equivalent prompt
following — and the defect suite found **no reproducible quality advantage
either** (counting, anatomy probes, symmetry, busy-ness all in the same band;
the INT8-penguin anecdote did not reproduce on any arm at fixed seeds). None
of the win conditions hold:
* not "several GB saved" (~1.1 GiB only);
* not faster (no offload existed to remove);
* not measurably better image semantics;
* Q5 ≈ Q4 speed here (both slower than fp8), so the Q5 preference rule is moot.

**Standing caveat:** if human review of the defect contact sheet finds Q4/Q5
visibly cleaner than the fp8 arm (e.g., typography), that would activate the
"equal/better semantics" clause and warrant a re-look — flagging rows 08
(HELLO SUN) and 02 (squirrel) as the places a difference would most plausibly
appear.

GGUF remains a **documented fallback** if VRAM headroom ever becomes the
binding constraint (e.g., co-resident VLM experiments): Q4_K_M at 9.6 GiB peak
frees ~1.1 GiB at a known ~11% (1024²) latency cost, reproducible from the
pins above.

## Reproducibility (GGUF arm, from scratch)

Comfy `7fe8a6138504f90ff7be82f3babf416da32876b1` ·
`git clone https://github.com/city96/ComfyUI-GGUF` →
`custom_nodes/ComfyUI-GGUF` @ `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` ·
`pip install gguf==0.17.1` · GGUFs into `ComfyUI/models/unet/` (sha256s above) ·
workflows `safe_image_proxy/workflows/bakeoff_klein_{q4_k_m,q5_k_m}.json` ·
bench `python bakeoff_bench.py --name <arm> --workflow <wf> --res 768,1024`,
VRAM pass `bakeoff_vram.py`, quality `bakeoff_quality.py` · launch unchanged
(`launch_comfy.py --listen 127.0.0.1 --port 8189`).

## Artifacts

* `benchmark_results/bakeoff/{baseline_env,fp8,gguf_q4_k_m,gguf_q5_k_m,vram_pass,quality_metrics,defect_manifest,defect_analysis}.json`
* Contact sheets: `output/contact_sheet_bakeoff_1024.png` (6 prompts × 3 arms),
  `output/contact_sheet_defect_suite.png` (16 defect probes × 3 arms — the
  quality-decision artifact for human review).
