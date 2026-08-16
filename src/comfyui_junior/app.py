import os
import time
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from comfyui_junior.config import settings
from comfyui_junior.safety import SafetyFilter, SafetyResult
from comfyui_junior.comfy import ComfyClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("comfyui_junior.app")

# Global singleton instances
safety_filter: Optional[SafetyFilter] = None
comfy_client: Optional[ComfyClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global safety_filter, comfy_client
    logger.info("Starting ComfyUI Junior service...")
    
    # 1. Initialize Safety Filter if enabled
    if settings.SAFETY_ENABLED:
        safety_path = Path(settings.SAFETY_MODEL_PATH)
        logger.info("Loading SafetyFilter from %s on %s...", safety_path, settings.SAFETY_DEVICE)
        try:
            safety_filter = SafetyFilter(model_dir=safety_path, device=settings.SAFETY_DEVICE)
        except Exception as e:
            logger.critical("FATAL: Failed to initialize SafetyFilter: %s", e)
            raise RuntimeError(f"SafetyFilter initialization failed: {e}")
    else:
        logger.warning("****************************************************************")
        logger.warning(" [WARNING] SAFETY_ENABLED=0: Running in UNPROTECTED BYPASS MODE! ")
        logger.warning("****************************************************************")
        
    # 2. Initialize Comfy Client
    try:
        comfy_client = ComfyClient(base_url=settings.comfy_base_url, workflow_path=settings.WORKFLOW_PATH)
        if comfy_client.check_health():
            logger.info("Successfully connected to ComfyUI backend at %s", settings.comfy_base_url)
        else:
            logger.warning("ComfyUI backend at %s is not yet reachable (will retry per request)", settings.comfy_base_url)
    except Exception as e:
        logger.critical("FATAL: Failed to initialize ComfyClient: %s", e)
        raise RuntimeError(f"ComfyClient initialization failed: {e}")
        
    yield
    logger.info("Shutting down ComfyUI Junior service...")

app = FastAPI(
    title="ComfyUI Junior",
    version="0.1.0",
    description="OpenAI-compatible safe image generation appliance powered by FLUX.2 Klein NVFP4",
    lifespan=lifespan
)

# Mount static files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_path = settings.STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ComfyUI Junior Appliance Running"}

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="A text description of the desired image(s).")
    model: Optional[str] = Field(default=settings.PUBLIC_MODEL_NAME, description="The model to use.")
    n: Optional[int] = Field(default=1, ge=1, le=1, description="Number of images to generate (currently fixed to 1).")
    size: Optional[str] = Field(default="1024x1024", description="Image resolution (e.g. 1024x1024, 768x1024).")
    response_format: Optional[str] = Field(default="b64_json", description="The format in which the generated images are returned.")
    user: Optional[str] = None

def openai_error_response(message: str, code: str, param: Optional[str] = None, status_code: int = 400, extra: Optional[dict] = None) -> JSONResponse:
    err_body = {
        "error": {
            "message": message,
            "type": "invalid_request_error" if status_code < 500 else "server_error",
            "param": param,
            "code": code
        }
    }
    if extra:
        err_body.update(extra)
    return JSONResponse(status_code=status_code, content=err_body)

@app.get("/health")
def health():
    backend_ok = comfy_client.check_health() if comfy_client else False
    return {
        "status": "ok" if backend_ok else "starting",
        "safety_enabled": settings.SAFETY_ENABLED,
        "comfy_backend_reachable": backend_ok,
        "public_model": settings.PUBLIC_MODEL_NAME
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.PUBLIC_MODEL_NAME,
                "object": "model",
                "created": 1770000000,
                "owned_by": "comfyui-junior"
            }
        ]
    }

@app.post("/v1/images/generations")
def generate_images(req: ImageGenerationRequest):
    # 1. Validate Model
    if req.model and req.model != settings.PUBLIC_MODEL_NAME:
        return openai_error_response(
            message=f"The model '{req.model}' does not exist. Supported model: '{settings.PUBLIC_MODEL_NAME}'",
            code="model_not_found",
            param="model",
            status_code=400
        )
        
    # 2. Validate Size
    try:
        parts = req.size.lower().split("x")
        if len(parts) != 2:
            raise ValueError()
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0 or width > 2048 or height > 2048:
            raise ValueError()
    except Exception:
        return openai_error_response(
            message=f"Invalid size '{req.size}'. Supported format: 'WIDTHxHEIGHT' (e.g. 1024x1024)",
            code="invalid_size",
            param="size",
            status_code=400
        )
        
    # 3. Inline Safety Filter
    if settings.SAFETY_ENABLED:
        if safety_filter is None:
            logger.error("SafetyFilter not initialized when SAFETY_ENABLED=1")
            return openai_error_response("Safety filter service unavailable", "server_error", status_code=500)
            
        try:
            res: SafetyResult = safety_filter.classify(req.prompt)
        except Exception as e:
            logger.error("Safety classification error: %s (failing closed)", e)
            return openai_error_response("Safety classification failed; request blocked", "server_error", status_code=500)
            
        logger.info("Safety result for '%s...': %s in %.1fms (reasons=%s)", req.prompt[:40], res.decision, res.latency_ms, res.reasons)
        
        if res.decision == "BLOCK":
            return openai_error_response(
                message=f"Prompt rejected by content safety policy: {', '.join(res.reasons) if res.reasons else 'prohibited content'}",
                code="content_policy_violation",
                param="prompt",
                status_code=400,
                extra={"safety_decision": "BLOCK", "reasons": res.reasons}
            )
        elif res.decision == "ROUTE":
            return openai_error_response(
                message=f"Prompt requires additional safety review: {', '.join(res.reasons)}",
                code="safety_route_required",
                param="prompt",
                status_code=400,
                extra={"safety_decision": "ROUTE", "reasons": res.reasons}
            )
    else:
        logger.warning("[SAFETY BYPASS] Generating prompt directly without safety check: '%s...'", req.prompt[:40])
        
    # 4. Submit to ComfyUI
    if comfy_client is None:
        return openai_error_response("ComfyUI client not initialized", "server_error", status_code=500)
        
    try:
        img_bytes, gen_latency = comfy_client.generate_image(
            prompt=req.prompt,
            width=width,
            height=height
        )
    except Exception as e:
        logger.error("Image generation backend failed: %s", e, exc_info=True)
        return openai_error_response("Image generation failed due to a backend error", "backend_error", status_code=502)
        
    b64_data = base64.b64encode(img_bytes).decode("utf-8")
    
    return {
        "created": int(time.time()),
        "data": [
            {
                "b64_json": b64_data,
                "revised_prompt": req.prompt
            }
        ]
    }
