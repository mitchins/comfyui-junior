"""Appliance supervisor: hardware contract, model provisioning, ComfyUI child.

Resolved duties:

1. enforce the stack-aware hardware contract (``blackwell_nvfp4`` requires
   SM120; ``ampere_fp8`` requires SM86 or newer);
2. resolve ``JUNIOR_IMAGE_STACK`` (auto-detecting from the GPU when unset) and
   export the resolved value so the application sees an explicit stack;
3. verify/provision the stack's model assets from the pinned manifest;
4. optionally enforce the PyTorch caching-allocator ceiling;
5. supervise the internal ComfyUI process and run the FastAPI application.
"""
import os
import sys
import time
import signal
import logging
import threading
import subprocess
import urllib.request
from pathlib import Path
import torch

from comfyui_junior.config import settings
from comfyui_junior.model_assets import ensure_model_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("comfyui_junior.launcher")


def detect_compute_capability():
    """Return the CUDA compute capability tuple, or ``None`` without CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    return None


def resolve_stack_for_process():
    """Resolve and export ``JUNIOR_IMAGE_STACK`` for this process.

    Auto-detection is resolved once here, so every later consumer (the app,
    the ComfyUI child) sees an explicit stack value via the environment.

    Returns:
        The resolved stack id (``blackwell_nvfp4`` or ``ampere_fp8``).
    """
    stack = settings.resolve_stack(detect_compute_capability())
    os.environ["JUNIOR_IMAGE_STACK"] = stack
    logger.info("Image stack resolved: %s", stack)
    return stack


def check_hardware_environment(stack: str):
    """Enforce the per-stack hardware contract.

    Args:
        stack: resolved stack id.

    Raises:
        RuntimeError: when no CUDA GPU is present, or the GPU's compute
            capability does not satisfy the stack (SM120 for
            ``blackwell_nvfp4``; SM86+ for ``ampere_fp8``).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("[Hardware Check Failed] No CUDA GPU detected. A supported NVIDIA GPU is required.")

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logger.info("Detected GPU: %s (Compute Capability: %d.%d, Total VRAM: %.2f GB)", device_name, major, minor, total_gb)

    if stack == "blackwell_nvfp4" and major != 12:
        raise RuntimeError(
            f"[Hardware Contract Violation] blackwell_nvfp4 requires an NVIDIA Blackwell SM120 GPU; "
            f"detected compute capability {major}.{minor}. Use JUNIOR_IMAGE_STACK=ampere_fp8 on this GPU."
        )
    if stack == "ampere_fp8" and (major < 8 or (major == 8 and minor < 6)):
        raise RuntimeError(
            f"[Hardware Contract Violation] ampere_fp8 requires SM86 (Ampere) or newer; "
            f"detected compute capability {major}.{minor}."
        )
    logger.info("Hardware contract satisfied for stack '%s'.", stack)


def effective_allocator_cap_gib(stack: str) -> float:
    """Return the allocator ceiling to enforce (0 disables the ceiling).

    Per-stack qualified defaults when the environment variable is unset:
    ``blackwell_nvfp4`` keeps the historical 10.0 GiB fence; ``ampere_fp8``
    runs uncapped (measured working set fits a 12 GB card with the API-side
    request-shape clamp — see docs/RTX3060_QUALIFICATION.md).
    """
    if os.getenv("COMFY_MEMORY_CAP_GIB") is not None:
        return float(os.environ["COMFY_MEMORY_CAP_GIB"])
    return 10.0 if stack == "blackwell_nvfp4" else 0.0


def apply_allocator_cap(cap_gib: float):
    """Enforce the PyTorch caching-allocator ceiling before model allocations."""
    if torch.cuda.is_available() and cap_gib > 0:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        target_bytes = cap_gib * (1024 ** 3)
        fraction = min(1.0, target_bytes / total_bytes)
        torch.cuda.memory.set_per_process_memory_fraction(fraction, 0)
        logger.info(
            "Enforced PyTorch caching-allocator ceiling: fraction=%.4f (%.2f GiB limit on %.2f GiB device)",
            fraction, cap_gib, total_bytes / (1024 ** 3)
        )
    else:
        logger.info("No PyTorch caching-allocator ceiling (cap=%.2f GiB)", cap_gib)


def wait_for_comfy_ready(base_url: str, timeout_seconds: float = 120.0) -> bool:
    """Poll the ComfyUI backend until it answers ``/system_stats``."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(f"{base_url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    """Run the appliance supervisor (hardware -> models -> ComfyUI -> app)."""
    logger.info("==================================================================")
    logger.info(" Starting ComfyUI Junior Appliance (v2.0.0)")
    logger.info("==================================================================")

    # 1. Resolve stack + enforce hardware contract
    stack = resolve_stack_for_process()
    check_hardware_environment(stack)

    # 2. Model Asset Preparation (stack-filtered)
    model_dir = Path(settings.MODEL_DIR)
    comfy_dir = Path(settings.COMFY_DIR)
    safety_model_path = Path(settings.SAFETY_MODEL_PATH)

    logger.info("Verifying model assets in %s (stack=%s)...", model_dir, stack)
    ensure_model_assets(
        model_dir=model_dir,
        comfy_dir=comfy_dir,
        hf_token=settings.HF_TOKEN,
        safety_model_path=safety_model_path,
        safety_hf_repo=os.getenv("SAFETY_HF_REPO") or None,
        safety_hf_revision=os.getenv("SAFETY_HF_REVISION") or None,
        stack=stack
    )

    # 3. Apply caching-allocator ceiling in this supervisor process
    cap_gib = effective_allocator_cap_gib(stack)
    apply_allocator_cap(cap_gib)

    # 4. Prepare ComfyUI launcher script that inherits the same ceiling
    comfy_script = f"""
import os
import sys
import torch
import logging

if torch.cuda.is_available() and {cap_gib} > 0:
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, ({cap_gib} * (1024**3)) / total_bytes)
    torch.cuda.memory.set_per_process_memory_fraction(fraction, 0)

sys.path.insert(0, '{settings.COMFY_DIR}')
os.chdir('{settings.COMFY_DIR}')
sys.argv = ['main.py', '--listen', '{settings.COMFY_HOST}', '--port', '{settings.COMFY_PORT}', '--disable-auto-launch', '--disable-all-custom-nodes']

import main
import app.logger
from app.assets.seeder import asset_seeder

event_loop, _, start_all_func = main.start_comfyui()
try:
    x = start_all_func()
    app.logger.print_startup_warnings()
    event_loop.run_until_complete(x)
except KeyboardInterrupt:
    logging.info("\\nStopped ComfyUI server")
finally:
    asset_seeder.shutdown()
    main.cleanup_temp()
"""
    comfy_env = os.environ.copy()
    comfy_proc = subprocess.Popen(
        [sys.executable, "-c", comfy_script],
        env=comfy_env
    )
    logger.info("Started internal ComfyUI process (PID: %d)", comfy_proc.pid)

    # 5. Wait for ComfyUI readiness
    logger.info("Awaiting internal ComfyUI readiness at %s...", settings.comfy_base_url)
    if not wait_for_comfy_ready(settings.comfy_base_url):
        logger.error("ComfyUI backend failed to start within timeout.")
        comfy_proc.terminate()
        sys.exit(1)
    logger.info("Internal ComfyUI is ready.")

    # 6. Start the Junior FastAPI application with process supervision
    import uvicorn
    config = uvicorn.Config(
        "comfyui_junior.app:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)

    def supervise_comfy():
        while not server.should_exit:
            ret = comfy_proc.poll()
            if ret is not None:
                logger.critical("Internal ComfyUI child process exited unexpectedly (code: %d)! Shutting down Junior...", ret)
                server.should_exit = True
                break
            time.sleep(0.5)

    monitor_thread = threading.Thread(target=supervise_comfy, daemon=True)
    monitor_thread.start()

    def shutdown_handler(signum, frame):
        logger.info("Received termination signal (%d), shutting down child processes...", signum)
        server.should_exit = True
        if comfy_proc.poll() is None:
            comfy_proc.terminate()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info("Starting Junior public service on %s:%d (role=%s)...",
                settings.HOST, settings.PORT,
                "evaluation" if settings.JUNIOR_EVALUATION_INSTANCE else "appliance")
    try:
        server.run()
    finally:
        if comfy_proc.poll() is None:
            logger.info("Terminating internal ComfyUI process...")
            comfy_proc.terminate()
            try:
                comfy_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                comfy_proc.kill()


if __name__ == "__main__":
    main()
