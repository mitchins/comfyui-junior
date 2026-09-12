import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved once at process start.

    Safety is NOT configurable: there is deliberately no enable/disable flag
    for the production prompt gates. The only bypass is the owner-operated
    evaluation instance (``JUNIOR_EVALUATION_INSTANCE``), a deliberate
    infrastructure-level decision that disables the frontend and marks the
    instance in ``/health``.
    """

    # Public Service
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    PUBLIC_MODEL_NAME: str = os.getenv("PUBLIC_MODEL_NAME", "flux2-klein-safe")

    # Internal ComfyUI Backend
    COMFY_HOST: str = os.getenv("COMFY_HOST", "127.0.0.1")
    COMFY_PORT: int = os.getenv("COMFY_PORT", "8188")
    COMFY_MEMORY_CAP_GIB: float = float(os.getenv("COMFY_MEMORY_CAP_GIB", "0") or 0)
    COMFY_DIR: str = os.getenv("COMFY_DIR", "/app/ComfyUI")

    # Image generation stack selection (hardware profile).
    #   blackwell_nvfp4 : SM120 native NVFP4 (RTX 5060 Ti class, 16 GB)
    #   ampere_fp8      : fp8 storage -> bf16 compute (RTX 3060 12 GB class)
    # Unset (default) auto-detects from the detected compute capability.
    JUNIOR_IMAGE_STACK: str = os.getenv("JUNIOR_IMAGE_STACK", "auto")

    # Storage Paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "/models")
    DATA_DIR: str = os.getenv("DATA_DIR", "/data")

    # Safety Classifier (unconditional in appliance mode)
    SAFETY_DEVICE: str = os.getenv("SAFETY_DEVICE", "cuda:0")
    SAFETY_MODEL_PATH: str = os.getenv("SAFETY_MODEL_PATH", "/models/safety/v29db")

    # Language envelope model (fastText lid.176)
    LID176_PATH: str = os.getenv("LID176_PATH", "/models/safety/lid.176.bin")

    # Hugging Face
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Owner-operated evaluation instance (infrastructure-level, deliberate).
    # When "1": prompt gates are bypassed by design, the child frontend is
    # disabled, and /health reports role=evaluation. Never expose such an
    # instance as the children's appliance.
    JUNIOR_EVALUATION_INSTANCE: bool = (
        os.getenv("JUNIOR_EVALUATION_INSTANCE", "0").strip() in ("1", "true", "yes")
    )

    # Packaged Asset Paths
    PACKAGE_ROOT: Path = Path(__file__).resolve().parent
    WORKFLOW_PATH_OVERRIDE: Optional[str] = os.getenv("WORKFLOW_PATH") or None

    def workflow_path(self, stack: str) -> Path:
        """Resolve the packaged workflow for a stack, honouring WORKFLOW_PATH.

        Args:
            stack: resolved stack id (``blackwell_nvfp4`` or ``ampere_fp8``).

        Returns:
            Path to the API-format ComfyUI workflow to execute.
        """
        if self.WORKFLOW_PATH_OVERRIDE:
            return Path(self.WORKFLOW_PATH_OVERRIDE)
        return self.PACKAGE_ROOT / "workflows" / self.STACK_WORKFLOWS[stack]

    # Quality presets: sampler steps per stack for "normal" and "high".
    QUALITY_STEPS: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "blackwell_nvfp4": {"normal": 4, "high": 8},
        "ampere_fp8": {"normal": 4, "high": 8},
    })

    # Workflow file per stack (relative to PACKAGE_ROOT/workflows).
    STACK_WORKFLOWS: Dict[str, str] = field(default_factory=lambda: {
        "blackwell_nvfp4": "flux2_klein_4b.json",
        "ampere_fp8": "flux2_klein_fp8_ampere.json",
    })

    @staticmethod
    def resolve_stack(detected_compute_capability: Optional[tuple] = None) -> str:
        """Resolve the image stack, auto-detecting when JUNIOR_IMAGE_STACK=auto.

        Args:
            detected_compute_capability: ``(major, minor)`` from CUDA, or
                ``None`` when detection should be skipped (unit tests).

        Returns:
            One of ``blackwell_nvfp4`` or ``ampere_fp8``.

        Raises:
            RuntimeError: on an unsupported GPU or an unknown explicit value.
        """
        requested = os.getenv("JUNIOR_IMAGE_STACK", "auto").strip().lower()
        if requested not in ("auto", "blackwell_nvfp4", "ampere_fp8"):
            raise RuntimeError(
                f"JUNIOR_IMAGE_STACK must be auto|blackwell_nvfp4|ampere_fp8 (got {requested!r})"
            )
        if requested != "auto":
            return requested
        if detected_compute_capability is None:
            raise RuntimeError("JUNIOR_IMAGE_STACK=auto but no GPU detected to auto-select a stack")
        major = detected_compute_capability[0]
        if major >= 12:
            return "blackwell_nvfp4"
        if major >= 8:
            return "ampere_fp8"
        raise RuntimeError(
            f"unsupported GPU compute capability {major}.{detected_compute_capability[1]} "
            "(need SM86+ for ampere_fp8 or SM120+ for blackwell_nvfp4)"
        )

    @property
    def comfy_base_url(self) -> str:
        """Base URL of the internal ComfyUI backend."""
        return f"http://{self.COMFY_HOST}:{self.COMFY_PORT}"


settings = Settings()
