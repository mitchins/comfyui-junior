import os
import re
import json
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("comfyui_junior.model_assets")

def calculate_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            sha.update(chunk)
    return sha.hexdigest()

def verify_file_sha256(filepath: Path, expected_sha256: str) -> bool:
    digest = calculate_sha256(filepath)
    if digest.lower() != expected_sha256.lower():
        logger.error("SHA256 mismatch for %s: expected %s, got %s", filepath, expected_sha256, digest)
        return False
    return True

def verify_directory_assets(
    dir_path: Path,
    expected_files: List[str],
    file_digests: Optional[Dict[str, str]] = None,
    verify_hashes: bool = False
) -> bool:
    if not dir_path.is_dir():
        logger.error("Expected directory at %s but found non-directory", dir_path)
        return False
    for rel_file in expected_files:
        fpath = dir_path / rel_file
        if not fpath.is_file() or fpath.stat().st_size == 0:
            logger.error("Directory asset %s missing expected file %s", dir_path, rel_file)
            return False
        if verify_hashes and file_digests and rel_file in file_digests:
            expected_digest = file_digests[rel_file]
            if not verify_file_sha256(fpath, expected_digest):
                logger.error("File %s in %s failed digest verification", rel_file, dir_path)
                return False
    return True

def load_manifest(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    if manifest_path is None:
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
    safety_hf_repo: Optional[str] = None,
    safety_hf_revision: Optional[str] = None,
    verify_hashes: bool = False,
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
        model_type = model_info.get("type")
        subfolder = model_info.get("subfolder", "")
        filename = model_info.get("filename", "")
        hf_repo = model_info.get("hf_repo")
        hf_filename = model_info.get("hf_filename")
        revision = model_info.get("revision")
        expected_sha256 = model_info.get("sha256")
        expected_files = model_info.get("files")
        file_digests = model_info.get("file_digests")
        is_optional = model_info.get("optional", False)

        # Allow override of safety model remote settings
        if model_type == "safety_classifier":
            if safety_hf_repo:
                if not safety_hf_revision or not re.fullmatch(r"[0-9a-fA-F]{40}", safety_hf_revision):
                    raise ValueError(
                        f"SAFETY_HF_REVISION must be an immutable 40-character hexadecimal commit SHA when SAFETY_HF_REPO is configured; got {safety_hf_revision!r}"
                    )
                hf_repo = safety_hf_repo
                revision = safety_hf_revision

        target_dir = model_dir / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename

        # Special case: safety model directory override
        if model_type == "safety_classifier" and safety_model_path:
            if safety_model_path.exists():
                if expected_files and not verify_directory_assets(
                    safety_model_path, expected_files, file_digests, verify_hashes=True
                ):
                    raise ValueError(f"Configured safety model at {safety_model_path} failed integrity checks.")
                results[model_id] = True
                logger.info("Safety model verified at configured path: %s", safety_model_path)
                continue

        # Check if model already exists
        if target_file.exists():
            if target_file.is_dir() and expected_files:
                check_hashes = True if model_type == "safety_classifier" else verify_hashes
                if verify_directory_assets(target_file, expected_files, file_digests, verify_hashes=check_hashes):
                    logger.info("Model directory '%s' valid at %s", model_id, target_file)
                    results[model_id] = True
                else:
                    logger.warning("Directory model '%s' at %s failed integrity checks; removing stale directory.", model_id, target_file)
                    shutil.rmtree(target_file, ignore_errors=True)
            elif target_file.is_file() and target_file.stat().st_size > 0:
                if verify_hashes and expected_sha256:
                    if not verify_file_sha256(target_file, expected_sha256):
                        raise ValueError(f"Integrity check failed for existing model '{model_id}' at {target_file}")
                logger.info("Model '%s' already present at %s", model_id, target_file)
                results[model_id] = True

        if not results.get(model_id, False):
            if dry_run:
                logger.info("[Dry Run] Model '%s' would be downloaded from %s (revision: %s)", model_id, hf_repo, revision)
                results[model_id] = False
                continue

            if not hf_repo:
                if is_optional:
                    logger.warning("Optional model '%s' not present locally and no HF repo provided.", model_id)
                    results[model_id] = False
                    continue
                else:
                    raise FileNotFoundError(f"Required model '{model_id}' missing and no HF repository defined.")

            logger.info("Downloading model '%s' from %s (revision: %s)...", model_id, hf_repo, revision)
            try:
                from huggingface_hub import hf_hub_download, snapshot_download

                if expected_files:
                    # Directory download with explicit pattern whitelist
                    snapshot_download(
                        repo_id=hf_repo,
                        revision=revision or None,
                        local_dir=str(target_file),
                        allow_patterns=expected_files,
                        token=hf_token or None
                    )
                    # Verify all expected files and digests are present
                    if not verify_directory_assets(target_file, expected_files, file_digests, verify_hashes=True):
                        shutil.rmtree(target_file, ignore_errors=True)
                        raise ValueError(f"Downloaded directory model '{model_id}' failed integrity validation at {target_file}")
                else:
                    # Single file download
                    downloaded_path = hf_hub_download(
                        repo_id=hf_repo,
                        filename=hf_filename or filename,
                        revision=revision or None,
                        token=hf_token or None
                    )
                    # Move/copy to target
                    if not target_file.exists():
                        shutil.copy2(downloaded_path, target_file)

                    if expected_sha256:
                        if not verify_file_sha256(target_file, expected_sha256):
                            target_file.unlink(missing_ok=True)
                            raise ValueError(f"Downloaded model '{model_id}' failed SHA-256 validation.")

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
            if comfy_target.is_symlink():
                try:
                    if comfy_target.resolve() != target_file.resolve():
                        comfy_target.unlink()
                        comfy_target.symlink_to(target_file)
                        logger.debug("Updated stale symlink: %s -> %s", comfy_target, target_file)
                except Exception as sym_err:
                    logger.warning("Failed to update symlink %s -> %s: %s", comfy_target, target_file, sym_err)
            elif not comfy_target.exists():
                try:
                    comfy_target.symlink_to(target_file)
                    logger.debug("Created symlink: %s -> %s", comfy_target, target_file)
                except Exception as sym_err:
                    logger.warning("Symlink creation failed for %s -> %s: %s", comfy_target, target_file, sym_err)

    return results
