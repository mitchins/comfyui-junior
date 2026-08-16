import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Settings:
    # Public Service
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    PUBLIC_MODEL_NAME: str = "flux2-klein-4b-safe"

    # Internal ComfyUI Backend
    COMFY_HOST: str = os.getenv("COMFY_HOST", "127.0.0.1")
    COMFY_PORT: int = int(os.getenv("COMFY_PORT", "8188"))
    COMFY_MEMORY_CAP_GIB: float = float(os.getenv("COMFY_MEMORY_CAP_GIB", "10.0"))
    COMFY_DIR: str = os.getenv("COMFY_DIR", "/app/ComfyUI")

    # Storage Paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "/models")
    DATA_DIR: str = os.getenv("DATA_DIR", "/data")

    # Safety Filter
    SAFETY_ENABLED: bool = os.getenv("SAFETY_ENABLED", "1").lower() in ("1", "true", "yes")
    SAFETY_DEVICE: str = os.getenv("SAFETY_DEVICE", "cuda:0")
    SAFETY_MODEL_PATH: str = os.getenv("SAFETY_MODEL_PATH", "/models/safety/v7_distilbert")
    SAFETY_HF_REPO: Optional[str] = os.getenv("SAFETY_HF_REPO") or None
    SAFETY_HF_REVISION: Optional[str] = os.getenv("SAFETY_HF_REVISION") or None

    # Hugging Face
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Packaged Asset Paths
    PACKAGE_ROOT: Path = Path(__file__).resolve().parent
    WORKFLOW_PATH: Path = Path(os.getenv("WORKFLOW_PATH", str(Path(__file__).resolve().parent / "workflows" / "flux2_klein_4b.json")))
    STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

    @property
    def comfy_base_url(self) -> str:
        return f"http://{self.COMFY_HOST}:{self.COMFY_PORT}"

settings = Settings()
