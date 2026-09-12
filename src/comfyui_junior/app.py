"""ComfyUI Junior public service: OpenAI-compatible API + Imagine frontend.

Request flow (appliance mode, unconditional):

    request -> style template (explicit allowlist)
            -> well-formedness gate        (prompt_format_invalid)
            -> English-envelope gate       (unsupported_language)
            -> FP16 v29db classifier       (content_policy_violation)
            -> policy_v6 PASS -> generation

No request field, header, or query parameter can weaken the prompt gates. The
classifier is a hard startup dependency; failures fail closed.

Evaluation instances (``JUNIOR_EVALUATION_INSTANCE=1``, an owner-operated
infrastructure-level decision made at process start) swap in the evaluation
bypass gate: generation is unfiltered by design so the owner can measure the
model externally, the child frontend is disabled, and ``/health`` reports
``role=evaluation``. The request-handler code path is identical in both
modes.
"""
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from comfyui_junior.classifier import JuniorSafetyClassifier
from comfyui_junior.comfy import ComfyClient
from comfyui_junior.config import settings
from comfyui_junior.pipeline import (
    FAILURE_LANGUAGE,
    FAILURE_POLICY,
    FAILURE_PROMPT_FORMAT,
    EvaluationBypassGate,
    GateResult,
    ProductionGate,
)
from comfyui_junior.styles import apply_style_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("comfyui_junior")

STATIC_DIR = settings.PACKAGE_ROOT / "static"

production_gate = None
comfy_client = None


def _detect_compute_capability():
    """Return the CUDA compute capability, or ``None`` in GPU-less contexts."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        return None
    return None


STACK = settings.resolve_stack(_detect_compute_capability())


def openai_error(message: str, code: str, param: Optional[str] = None,
                 status_code: int = 400, extra: Optional[dict] = None) -> JSONResponse:
    """Build an OpenAI-style error response.

    Args:
        message: human-readable message shown to clients.
        code: machine-readable error code.
        param: offending request parameter, when known.
        status_code: HTTP status code.
        extra: additional top-level fields (e.g. safety decision details).

    Returns:
        A ``JSONResponse`` in the OpenAI error envelope shape.
    """
    body = {"error": {
        "message": message,
        "type": "invalid_request_error" if status_code < 500 else "server_error",
        "param": param, "code": code,
    }}
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: resolve the stack, load the gate and backend client.

    In appliance mode the FP16 v29db classifier is a hard dependency: if it
    cannot load on CUDA the service refuses to start. Evaluation instances
    inject the deliberate bypass gate instead and disable the frontend.
    """
    global production_gate, comfy_client

    if settings.JUNIOR_EVALUATION_INSTANCE:
        production_gate = EvaluationBypassGate()
        logger.warning("=" * 64)
        logger.warning(" EVALUATION INSTANCE: prompt filtering DISABLED by owner")
        logger.warning(" configuration (JUNIOR_EVALUATION_INSTANCE=1). This")
        logger.warning(" process is an unfiltered measurement endpoint and must")
        logger.warning(" never be exposed as the children's appliance.")
        logger.warning("=" * 64)
    else:
        classifier = JuniorSafetyClassifier(settings.SAFETY_MODEL_PATH, device=settings.SAFETY_DEVICE)
        if str(classifier.dtype) != "torch.float16" or "cuda" not in str(classifier.device):
            raise RuntimeError("production classifier must be FP16 on CUDA — refusing to start")
        production_gate = ProductionGate(classifier)

    try:
        comfy_client = ComfyClient(
            base_url=settings.comfy_base_url,
            workflow_path=str(settings.workflow_path(STACK)),
            quality_steps=settings.QUALITY_STEPS.get(STACK),
        )
        if comfy_client.check_health():
            logger.info("Connected to ComfyUI backend at %s (stack=%s)", settings.comfy_base_url, STACK)
        else:
            logger.warning("ComfyUI backend not reachable yet (will retry per request)")
    except Exception as e:
        logger.critical("FATAL: ComfyClient init failed: %s", e)
        raise

    # Warm the language gate so the first request does not pay load cost.
    from comfyui_junior import langgate
    try:
        langgate.check("a warm up prompt for the language gate")
    except Exception:
        logger.warning("language gate warm-up skipped (lid.176 not present yet)")

    logger.info("comfyui_junior up (role=%s, stack=%s)",
                "evaluation" if settings.JUNIOR_EVALUATION_INSTANCE else "appliance", STACK)
    yield
    logger.info("shutting down comfyui_junior")


app = FastAPI(title="ComfyUI Junior - Safe Image Proxy", version="2.0.0", lifespan=lifespan)

if not settings.JUNIOR_EVALUATION_INSTANCE and STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the Imagine studio (appliance) or an evaluation notice (eval)."""
    if settings.JUNIOR_EVALUATION_INSTANCE:
        return JSONResponse({
            "role": "evaluation",
            "message": "Evaluation instance: prompt filtering is disabled by owner "
                       "configuration and the frontend is unavailable. Use "
                       "POST /v1/images/generations for unfiltered measurement.",
        })
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ComfyUI Junior API running"}


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request.

    NOTE: no field can influence safety. ``style`` selects an explicit,
    fixed template from a server-side allowlist (additive-only suffix); it
    cannot alter gate behaviour.
    """

    model: Optional[str] = Field(default=settings.PUBLIC_MODEL_NAME)
    n: Optional[int] = Field(default=1, ge=1, le=1)
    prompt: str = Field(..., min_length=1, max_length=1500)
    quality: Optional[str] = Field(default="normal", description='"normal" or "high"')
    response_format: Optional[str] = Field(default="b64_json", description='Only "b64_json" is supported.')
    size: Optional[str] = Field(default="1024x1024")
    style: Optional[str] = Field(default="none", description='"none" or a named template, e.g. "colouring_sheet"')
    user: Optional[str] = None


@app.get("/health")
def health():
    """Report service health, role, safety pipeline and backend reachability."""
    return {
        "status": "ok",
        "role": "evaluation" if settings.JUNIOR_EVALUATION_INSTANCE else "appliance",
        "gates": "disabled-by-owner (JUNIOR_EVALUATION_INSTANCE=1)"
                 if settings.JUNIOR_EVALUATION_INSTANCE
                 else "wellformed -> language_envelope -> v29db_fp16 -> policy_v6",
        "classifier_dtype": str(production_gate.classifier.dtype) if production_gate and hasattr(production_gate, "classifier") else None,
        "policy": "policy_v6" if not settings.JUNIOR_EVALUATION_INSTANCE else "evaluation-bypass",
        "stack": STACK,
        "comfy_backend_reachable": comfy_client.check_health() if comfy_client else False,
        "public_model": settings.PUBLIC_MODEL_NAME,
    }


@app.get("/v1/models")
def list_models():
    """List the single public model id in the OpenAI listing shape."""
    return {"object": "list", "data": [{
        "id": settings.PUBLIC_MODEL_NAME,
        "object": "model",
        "created": 1770000000,
        "owned_by": "comfy-appliance",
    }]}


@app.post("/v1/images/generations")
def generate_images(req: ImageGenerationRequest):
    """Generate one image after validation and the prompt-safety pipeline."""
    # 1. model / size / quality / style validation (none of this touches safety)
    if req.model and req.model != settings.PUBLIC_MODEL_NAME:
        return openai_error(
            f"The model '{req.model}' does not exist. Supported model: '{settings.PUBLIC_MODEL_NAME}'",
            "model_not_found", "model", 400)
    if req.response_format not in (None, "b64_json"):
        return openai_error("Only response_format='b64_json' is supported.",
                            "invalid_response_format", "response_format", 400)
    try:
        parts = req.size.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        width, height = int(parts[0]), int(parts[1])
        # Product surface is 16..1344 per side and a multiple of 16 (the
        # frontend offers 768/1024/1344). Larger latents are rejected at the
        # API boundary so the measured GPU working set cannot be exceeded by
        # request shape.
        if not (16 <= width <= 1344 and 16 <= height <= 1344) or width % 16 or height % 16:
            raise ValueError
    except ValueError:
        return openai_error(
            "Invalid size. Supported format: 'WIDTHxHEIGHT' with each side a "
            "multiple of 16 between 16 and 1344 (e.g. 1024x1024).",
            "invalid_size", "size", 400)
    if req.quality not in ("normal", "high"):
        return openai_error("quality must be 'normal' or 'high'", "invalid_quality", "quality", 400)

    # 2. Explicit style template (fixed allowlist; no implicit rewriting).
    #    Expanded BEFORE the gates so the classifier judges the exact text
    #    that will be rendered.
    try:
        render_prompt, style_template = apply_style_template(req.prompt, req.style)
    except ValueError as e:
        return openai_error(str(e), "invalid_style", "style", 400)
    if style_template:
        logger.info("style template applied (%s)", style_template)
    steps = settings.QUALITY_STEPS.get(STACK, {}).get(req.quality)

    # 3. UNCONDITIONAL prompt-safety pipeline (or the evaluation bypass on
    #    evaluation instances). No flag, parameter, header, or request field
    #    can skip or weaken the appliance-mode gates.
    if production_gate is None:
        return openai_error("Safety pipeline unavailable; request refused", "server_error", status_code=500)
    result: GateResult = production_gate.check(render_prompt)
    if not result.ok:
        friendly = {
            FAILURE_PROMPT_FORMAT: "Hmm, we couldn't read that. Please write your idea in normal words and letters.",
            FAILURE_LANGUAGE: "Sorry! We can only understand English right now. Try writing your idea in English.",
            FAILURE_POLICY: "That idea isn't something we can make a picture of. Try a different, friendlier idea!",
        }.get(result.failure_code, "We can't use that prompt.")
        logger.info("gate rejected (code=%s gate=%s): %.60r", result.failure_code, result.gate, req.prompt)
        return openai_error(friendly, result.failure_code or FAILURE_POLICY, "prompt", 400,
                            extra={"safety_failure_code": result.failure_code, "gate": result.gate})
    logger.info("gate %s (%.1fms): %.60r", result.gate,
                result.classification.latency_ms if result.classification else -1, render_prompt)

    # 4. Generation.
    if comfy_client is None:
        return openai_error("Backend client unavailable", "server_error", status_code=500)
    try:
        img_bytes, gen_latency = comfy_client.generate_image(
            prompt=render_prompt, width=width, height=height, steps=steps)
    except Exception as e:
        logger.error("backend generation failed: %s", e)
        return openai_error(f"Backend generation failed: {e}", "backend_error", status_code=502)

    return {
        "created": int(time.time()),
        "data": [{
            "b64_json": base64.b64encode(img_bytes).decode("utf-8"),
            "revised_prompt": render_prompt,
        }],
        "meta": {"generation_latency_s": round(gen_latency, 3), "quality": req.quality,
                 "style_template": style_template},
    }
