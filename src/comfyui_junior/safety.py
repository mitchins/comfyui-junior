import os
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel

logger = logging.getLogger("comfyui_junior.safety")

@dataclass
class SafetyResult:
    decision: Literal["PASS", "ROUTE", "BLOCK"]
    reasons: List[str]
    scores: Dict[str, List[float]]
    latency_ms: float

class SafetyFilter:
    """
    Prompt safety classifier using DistilBERT encoder and trained multi-width regression heads.
    """
    def __init__(self, model_dir: Path, device: str = "cuda:0"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        logger.info("Initializing SafetyFilter from %s on %s...", self.model_dir, self.device)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Safety model directory not found: {self.model_dir}")
            
        heads_path = self.model_dir / "heads.pt"
        if not heads_path.exists():
            raise FileNotFoundError(f"Safety heads file not found: {heads_path}")
            
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.encoder = DistilBertModel.from_pretrained(str(self.model_dir)).to(self.device)
        self.encoder.eval()
        
        h = torch.load(str(heads_path), map_location=self.device)
        self.dims: List[str] = h["dims"]
        self.widths: List[int] = h["widths"]
        
        self.heads = nn.ModuleDict()
        for i, (dim, width) in enumerate(zip(self.dims, self.widths)):
            lin = nn.Linear(768, width)
            lin.weight.data = h["heads"][f"{i}.weight"].to(self.device)
            lin.bias.data = h["heads"][f"{i}.bias"].to(self.device)
            self.heads[dim] = lin
        self.heads.eval()
        
        logger.info("SafetyFilter initialized in %.2fs (dims=%s, widths=%s)", time.time() - t0, self.dims, self.widths)

    def classify(self, prompt: str) -> SafetyResult:
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            out = self.encoder(**inputs)
            cls_rep = out.last_hidden_state[:, 0]
            
            scores: Dict[str, List[float]] = {}
            for dim in self.dims:
                raw = self.heads[dim](cls_rep).squeeze(0).cpu().tolist()
                scores[dim] = [raw] if isinstance(raw, float) else raw
                
        decision: Literal["PASS", "ROUTE", "BLOCK"] = "PASS"
        reasons: List[str] = []
        s = scores
        
        # Policy Evaluation for Child Safety (~12yo Appliance)
        # 1. Hard BLOCK Criteria
        # sexual (3 widths: mild, overt, explicit)
        if s["sexual"][2] >= 0.0 or s["sexual"][1] >= 0.5 or s["sexual"][0] >= 2.0:
            decision = "BLOCK"
            reasons.append("sexual_content")
            
        # nudity (2 widths: partial, full)
        if s["nudity"][1] >= 0.5 or s["nudity"][0] >= 2.5:
            decision = "BLOCK"
            reasons.append("nudity")
            
        # violence_gore (2 widths: combat, gore)
        if s["violence_gore"][1] >= 0.5 or s["violence_gore"][0] >= 1.5:
            decision = "BLOCK"
            reasons.append("violence_gore")
            
        # substances (1 width: drugs)
        if s["substances"][0] >= 1.0:
            decision = "BLOCK"
            reasons.append("substances")
            
        # disturbing (2 widths: macabre, grotesque)
        if s["disturbing"][1] >= 0.5 or s["disturbing"][0] >= 1.5:
            decision = "BLOCK"
            reasons.append("disturbing_content")
            
        # fetish (2 widths: mild, severe)
        if s["fetish"][1] >= 0.5 or s["fetish"][0] >= 1.5:
            decision = "BLOCK"
            reasons.append("fetish_content")
            
        # 2. ROUTE Criteria (Borderline/suggestive content not reaching BLOCK)
        if decision == "PASS":
            route_reasons = []
            for dim, vals in s.items():
                if any(v >= 0.0 for v in vals):
                    route_reasons.append(dim)
            if route_reasons:
                decision = "ROUTE"
                reasons = route_reasons
                
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SafetyResult(decision=decision, reasons=reasons, scores=scores, latency_ms=round(latency_ms, 2))
