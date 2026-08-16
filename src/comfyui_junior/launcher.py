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

def check_hardware_environment():
    """
    Validates that a supported NVIDIA Blackwell GPU (SM120) is present.
    Enforces the v1 hardware contract.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("[Hardware Check Failed] No CUDA GPU detected. Appliance requires an NVIDIA Blackwell GPU (SM120, CC 12.0) for native NVFP4 execution.")
        
    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    logger.info("Detected GPU: %s (Compute Capability: %d.%d, Total VRAM: %.2f GB)", device_name, major, minor, total_gb)
    
    if major != 12:
        raise RuntimeError(
            f"[Hardware Contract Violation] Device compute capability is {major}.{minor}. "
            "ComfyUI Junior v1 strictly requires an NVIDIA Blackwell SM120 GPU (Compute Capability 12.0) for native NVFP4 express path."
        )
    logger.info("Blackwell SM120 architecture confirmed. Native NVFP4 fast-path active.")

def apply_allocator_cap():
    """
    Enforces PyTorch caching-allocator ceiling before model allocations.
    """
    if torch.cuda.is_available() and settings.COMFY_MEMORY_CAP_GIB > 0:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        target_bytes = settings.COMFY_MEMORY_CAP_GIB * (1024 ** 3)
        fraction = min(1.0, target_bytes / total_bytes)
        torch.cuda.memory.set_per_process_memory_fraction(fraction, 0)
        logger.info(
            "Enforced PyTorch caching-allocator ceiling: fraction=%.4f (%.2f GiB limit on %.2f GiB device)",
            fraction, settings.COMFY_MEMORY_CAP_GIB, total_bytes / (1024 ** 3)
        )

def wait_for_comfy_ready(base_url: str, timeout_seconds: float = 60.0) -> bool:
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
    logger.info("==================================================================")
    logger.info(" Starting ComfyUI Junior Appliance (v0.1.0)")
    logger.info("==================================================================")

    # 1. Enforce Hardware Contract
    check_hardware_environment()

    # 2. Model Asset Preparation
    model_dir = Path(settings.MODEL_DIR)
    comfy_dir = Path(settings.COMFY_DIR)
    safety_model_path = Path(settings.SAFETY_MODEL_PATH)
    
    logger.info("Verifying model assets in %s...", model_dir)
    ensure_model_assets(
        model_dir=model_dir,
        comfy_dir=comfy_dir,
        hf_token=settings.HF_TOKEN,
        safety_model_path=safety_model_path,
        safety_hf_repo=settings.SAFETY_HF_REPO,
        safety_hf_revision=settings.SAFETY_HF_REVISION
    )

    # 3. Apply Caching-Allocator Ceiling in this supervisor process
    apply_allocator_cap()

    # 4. Prepare ComfyUI launcher script that inherits allocator cap
    comfy_script = f"""
import os
import sys
import torch
import logging

if torch.cuda.is_available():
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, ({settings.COMFY_MEMORY_CAP_GIB} * (1024**3)) / total_bytes)
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

    # 5. Wait for ComfyUI Readiness
    logger.info("Awaiting internal ComfyUI readiness at %s...", settings.comfy_base_url)
    if not wait_for_comfy_ready(settings.comfy_base_url, timeout_seconds=120.0):
        logger.error("ComfyUI backend failed to start within timeout.")
        comfy_proc.terminate()
        sys.exit(1)
    logger.info("Internal ComfyUI is ready.")

    # 6. Start Junior FastAPI Proxy with Process Supervision
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

    logger.info("Starting Junior public service on %s:%d...", settings.HOST, settings.PORT)
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
