import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("comfyui_junior.model_assets")

def load_manifest(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    if manifest_path is None:
        # Check standard config path
        candidate = Path("/app/config/models.json")
        if not candidate.exists():
            candidate = Path(__file__).resolve().parent.parent.parent / "config" / "models.json"
        manifest_path = candidate

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}

def ensure_model_assets(
    model_dir: Path,
    comfy_dir: Path,
    hf_token: Optional[str] = None,
    safety_model_path: Optional[Path] = None,
    dry_run: bool = False
) -> Dict[str, bool]:
    """
    Checks and idempotently ensures all required models exist on disk.
    If comfy_dir has a models/ folder, ensures symlinks/paths are aligned.
    """
    manifest = load_manifest()
    models = manifest.get("models", [])
    results: Dict[str, bool] = {}

    model_dir.mkdir(parents=True, exist_ok=True)
    comfy_models_root = comfy_dir / "models"

    for model_info in models:
        model_id = model_info.get("id")
        subfolder = model_info.get("subfolder", "")
        filename = model_info.get("filename", "")
        hf_repo = model_info.get("hf_repo")
        hf_filename = model_info.get("hf_filename")
        is_optional = model_info.get("optional", False)

        target_dir = model_dir / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename

        # Special case: safety model directory override
        if model_info.get("type") == "safety_classifier" and safety_model_path:
            if safety_model_path.exists():
                results[model_id] = True
                logger.info("Safety model found at configured path: %s", safety_model_path)
                continue

        # Check if model already exists
        if target_file.exists() and (target_file.is_dir() or target_file.stat().st_size > 0):
            logger.info("Model '%s' already present at %s", model_id, target_file)
            results[model_id] = True
        else:
            if dry_run:
                logger.info("[Dry Run] Model '%s' would be downloaded from %s", model_id, hf_repo)
                results[model_id] = False
                continue

            if not hf_repo:
                if is_optional:
                    logger.warning("Optional model '%s' not present locally and no HF repo provided.", model_id)
                    results[model_id] = False
                    continue
                else:
                    raise FileNotFoundError(f"Required model '{model_id}' missing and no HF repository defined.")

            logger.info("Downloading model '%s' from %s...", model_id, hf_repo)
            try:
                from huggingface_hub import hf_hub_download, snapshot_download

                if "files" in model_info:
                    # Directory download
                    snapshot_download(
                        repo_id=hf_repo,
                        local_dir=str(target_file),
                        token=hf_token or None
                    )
                else:
                    # Single file download
                    downloaded_path = hf_hub_download(
                        repo_id=hf_repo,
                        filename=hf_filename or filename,
                        token=hf_token or None
                    )
                    # Move/link to target
                    if not target_file.exists():
                        import shutil
                        shutil.copy2(downloaded_path, target_file)

                results[model_id] = True
                logger.info("Successfully acquired model '%s'", model_id)
            except Exception as e:
                if is_optional:
                    logger.warning("Optional model '%s' download skipped/failed: %s", model_id, e)
                    results[model_id] = False
                else:
                    logger.error("Failed to download required model '%s': %s", model_id, e)
                    raise

        # Ensure ComfyUI models directory points to the model
        if comfy_models_root.exists() and subfolder and target_file.exists():
            comfy_subfolder = comfy_models_root / subfolder
            comfy_subfolder.mkdir(parents=True, exist_ok=True)
            comfy_target = comfy_subfolder / filename
            if not comfy_target.exists():
                try:
                    comfy_target.symlink_to(target_file)
                    logger.debug("Created symlink: %s -> %s", comfy_target, target_file)
                except Exception as sym_err:
                    logger.debug("Symlink creation skipped: %s", sym_err)

    return results
