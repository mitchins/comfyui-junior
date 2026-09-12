"""Production prompt-safety classifier: v29db DeBERTa-v3-base, FP16, single-view.

Frozen artifact: ``Mitchins/comfyui-junior-safety`` at revision
``a377d25017065db0ec5b8a7b6a2d11a30a906d5b`` (provisioned and digest-verified
by :mod:`comfyui_junior.model_assets`).

Serving profile (per the published model card and the v29 ship rationale):

* **Single-view**: the raw prompt only. The published two-view canonicalised
  inference remains inside the frozen HF artifact for parity/regression use;
  production instead requires well-formed input (``wellformed``) and an
  English envelope (``langgate``) *before* classification.
* **FP16 on CUDA**. No quantisation, no BF16, no CPU serving (CPU instances
  are permitted only for diagnostics and are refused by the application).
* Architecture: 12-layer DeBERTa-v3 encoder (hidden 768), masked mean pooling
  over ``last_hidden_state``, six cumulative-logit ordinal heads with widths
  ``[3, 2, 2, 2, 2, 2]``; the decoded level is the count of leading sigmoid
  thresholds above 0.5.
* ``policy_v6`` binary PASS/BLOCK via :data:`BLOCK_AT`.

No ``trust_remote_code``; weights load from the pinned local snapshot.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("comfyui_junior.classifier")

#: Dimension names in checkpoint order (schema is verified at load).
DIMS: List[str] = ["sexual", "nudity", "violence_gore", "substances", "disturbing", "fetish"]

#: Ordinal head widths in the same order (schema is verified at load).
WIDTHS: List[int] = [3, 2, 2, 2, 2, 2]

#: Maximum tokenizer sequence length used by the frozen artifact.
MAX_LEN: int = 256

#: Frozen policy identifier reported through the API surface.
POLICY_VERSION: str = "policy_v6"

#: Frozen upstream revision this artifact was qualified from.
MODEL_REVISION: str = "a377d25017065db0ec5b8a7b6a2d11a30a906d5b"

#: policy_v6 block thresholds per dimension (level >= threshold -> BLOCK).
BLOCK_AT: Dict[str, int] = {
    "sexual": 2,
    "nudity": 2,
    "violence_gore": 2,
    "substances": 2,
    "disturbing": 2,
    "fetish": 1,
}


@dataclass
class ClassificationResult:
    """Outcome of classifying one prompt against policy_v6.

    Attributes mirror the frozen artifact's reference output: decoded ordinal
    ``levels`` per dimension, per-threshold sigmoid ``probabilities``, the
    binary ``decision`` and human-readable ``reasons`` for BLOCK results.
    """

    decision: Literal["PASS", "BLOCK"]
    levels: Dict[str, int]
    probabilities: Dict[str, List[float]]
    reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    policy: str = POLICY_VERSION


#: Manifest record id for the production safety artifact.
_MANIFEST_ID = "junior-safety-v29db"


def _verify_artifact_against_manifest(model_dir: str) -> None:
    """Fail closed unless the artifact matches the pinned manifest digests.

    Defence-in-depth against tampered or stale weights after provisioning:
    every file listed in the ``junior-safety-v29db`` manifest record must
    exist with the exact pinned SHA-256, and no verification may be skipped.

    Args:
        model_dir: directory holding the served classifier artifact.

    Raises:
        ValueError: when the manifest record is missing, incomplete, or any
            file digest deviates from the pin.
    """
    import hashlib

    from comfyui_junior.model_assets import load_manifest

    records = [m for m in load_manifest().get("models", []) if m.get("id") == _MANIFEST_ID]
    if len(records) != 1:
        raise ValueError(f"expected exactly one '{_MANIFEST_ID}' model record in the manifest")
    record = records[0]
    expected_files = record.get("files") or []
    digests = record.get("file_digests") or {}
    # First pass: every expected file must have a pinned digest.
    for rel in expected_files:
        if rel not in digests:
            raise ValueError(f"missing digest for expected file '{rel}'")
    # Second pass: every file must exist with the exact pinned digest.
    for rel in expected_files:
        path = os.path.join(model_dir, rel)
        if not os.path.isfile(path):
            raise ValueError(f"expected artifact file '{rel}' missing from {model_dir}")
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        if sha.hexdigest().lower() != digests[rel].lower():
            raise ValueError(f"artifact file '{rel}' failed integrity verification")


class JuniorSafetyClassifier:
    """FP16 single-view v29db classifier used by the production prompt gate."""

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        """Load the pinned v29db artifact and convert it to FP16 on CUDA.

        Args:
            model_dir: directory holding the digest-verified snapshot
                (``model.safetensors``, ``heads.pt``, tokenizer files).
            device: requested device; anything that is not an available CUDA
                device falls back to CPU with a diagnostic-only warning.

        Raises:
            FileNotFoundError: if the artifact directory or ``heads.pt`` is missing.
            ValueError: if the checkpoint schema or encoder geometry deviates
                from the frozen artifact, or if a CUDA deployment is not FP16.
        """
        self.model_dir = model_dir
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"classifier artifact dir not found: {model_dir}")
        heads_path = os.path.join(model_dir, "heads.pt")
        if not os.path.exists(heads_path):
            raise FileNotFoundError(f"heads.pt missing in {model_dir}")
        _verify_artifact_against_manifest(model_dir)

        requested = torch.device(device)
        self.device = requested if torch.cuda.is_available() and requested.type == "cuda" else torch.device("cpu")
        if self.device.type != "cuda":
            logger.warning("classifier running on CPU — diagnostic use only")

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # FP32 load then explicit round-to-nearest FP16 cast, so the serving
        # weights are bit-derived from the same source as the FP32 reference.
        encoder = AutoModel.from_pretrained(model_dir, dtype=torch.float32)
        self.encoder = encoder.to(device=self.device, dtype=torch.float16).eval()

        ck = torch.load(heads_path, map_location="cpu", weights_only=True)
        dims, widths = list(ck["dims"]), list(ck["widths"])
        if dims != DIMS or widths != WIDTHS:
            raise ValueError(f"checkpoint schema mismatch: dims={dims} widths={widths} — refusing to serve")
        hidden = self.encoder.config.hidden_size
        if hidden != 768 or self.encoder.config.num_hidden_layers != 12:
            raise ValueError(
                f"unexpected encoder geometry: hidden={hidden} layers={self.encoder.config.num_hidden_layers} — refusing to serve"
            )
        self.heads = nn.ModuleList([nn.Linear(hidden, w) for w in widths])
        self.heads.load_state_dict(ck["heads"])
        self.heads = self.heads.to(device=self.device, dtype=torch.float16).eval()

        self.cold_load_s = time.perf_counter() - t0
        actual_dtype = next(self.parameters()).dtype
        if self.device.type == "cuda" and actual_dtype != torch.float16:
            raise RuntimeError(f"classifier is not FP16 on CUDA (dtype={actual_dtype}) — refusing to serve")
        logger.info(
            "v29db classifier loaded from %s on %s in %.2fs (dtype=%s)",
            model_dir, self.device, self.cold_load_s, actual_dtype,
        )

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield all encoder and head parameters (used for dtype verification)."""
        yield from self.encoder.parameters()
        yield from self.heads.parameters()

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Dtype of the served parameters (``torch.float16`` when qualified)."""
        return next(self.parameters()).dtype

    @torch.no_grad()
    def _pooled(self, texts: List[str]) -> torch.Tensor:
        """Encode texts and apply masked mean pooling over the sequence."""
        enc = self.tokenizer(
            texts, truncation=True, max_length=MAX_LEN,
            padding=True, return_tensors="pt",
        ).to(self.device)
        out = self.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def classify_batch(self, prompts: List[str], batch_size: int = 16) -> List[ClassificationResult]:
        """Classify a batch of prompts against policy_v6.

        Args:
            prompts: raw prompt strings (single-view; no canonicalisation).
            batch_size: encoder batch size for chunked inference.

        Returns:
            One :class:`ClassificationResult` per prompt, in input order.
        """
        results: List[ClassificationResult] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            pooled = self._pooled(chunk)
            logits = [h(pooled) for h in self.heads]
            # Sigmoid in FP32; decoding is thresholded so tiny probability
            # drift cannot flip a level without exceeding the 0.5 threshold.
            probs = [torch.sigmoid(lg.float()).cpu() for lg in logits]
            for j in range(len(chunk)):
                results.append(self._decode([p[j].tolist() for p in probs]))
        return results

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a single prompt and record wall-clock latency."""
        t0 = time.perf_counter()
        res = self.classify_batch([prompt])[0]
        res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return res

    @staticmethod
    def _decode(per_dim_probs: List[List[float]]) -> ClassificationResult:
        """Decode per-threshold sigmoid probabilities into ordinal levels and a decision."""
        levels: Dict[str, int] = {}
        probabilities: Dict[str, List[float]] = {}
        for dim, pk in zip(DIMS, per_dim_probs):
            probabilities[dim] = [round(x, 6) for x in pk]
            lvl = 0
            for v in pk:  # cumulative-logit ordinal: count the leading run above 0.5
                if v > 0.5:
                    lvl += 1
                else:
                    break
            levels[dim] = lvl
        hits = [d for d in DIMS if levels[d] >= BLOCK_AT[d]]
        return ClassificationResult(
            decision="BLOCK" if hits else "PASS",
            levels=levels,
            probabilities=probabilities,
            reasons=[f"{d} level {levels[d]} >= {BLOCK_AT[d]}" for d in hits],
        )
