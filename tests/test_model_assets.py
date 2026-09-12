"""Tests for model provisioning, manifest integrity and workflow validation."""
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from comfyui_junior.comfy import ComfyClient
from comfyui_junior.model_assets import (
    calculate_sha256,
    ensure_model_assets,
    load_manifest,
    verify_file_sha256,
)

WORKFLOWS = Path(__file__).resolve().parent.parent / "src" / "comfyui_junior" / "workflows"


class TestModelAssets(unittest.TestCase):
    """Manifest structure, stack filtering and digest helpers."""

    def test_load_manifest(self):
        """The v2 manifest contains both stacks and every pinned asset."""
        manifest = load_manifest()
        self.assertIn("models", manifest)
        models = manifest["models"]
        ids = [m["id"] for m in models]
        for expected in ("flux2-klein-4b-nvfp4", "flux2-klein-4b-fp8",
                         "qwen3-4b-fp4-flux2", "qwen3-4b-fp8-mixed",
                         "flux2-vae", "junior-safety-v29db", "fasttext-lid176"):
            self.assertIn(expected, ids)

        for m in models:
            if m.get("optional"):
                continue
            if m.get("type") == "safety_classifier":
                self.assertTrue(m.get("file_digests"))
            else:
                self.assertIsNotNone(m.get("sha256") or m.get("url"))

    def test_stack_field_partitioning(self):
        """Stack-specific entries are tagged; shared entries carry no stacks field."""
        manifest = load_manifest()
        by_id = {m["id"]: m for m in manifest["models"]}
        self.assertEqual(by_id["flux2-klein-4b-nvfp4"]["stacks"], ["blackwell_nvfp4"])
        self.assertEqual(by_id["flux2-klein-4b-fp8"]["stacks"], ["ampere_fp8"])
        for shared in ("flux2-vae", "junior-safety-v29db", "fasttext-lid176"):
            self.assertNotIn("stacks", by_id[shared])

    def test_ensure_model_assets_dry_run_stack_filtered(self):
        """Dry-run reports only the requested stack's entries as pending."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = ensure_model_assets(model_dir=Path(tmp_dir) / "models",
                                      comfy_dir=Path(tmp_dir) / "comfy",
                                      dry_run=True, stack="ampere_fp8")
            self.assertIn("flux2-klein-4b-fp8", res)
            self.assertFalse(res["flux2-klein-4b-fp8"])
            self.assertIn("fasttext-lid176", res)
            self.assertNotIn("flux2-klein-4b-nvfp4", res)
            self.assertNotIn("qwen3-4b-fp4-flux2", res)

    def test_sha256_verification(self):
        """The digest helper verifies known content and rejects mismatches."""
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"ComfyUI Junior Test Content")
            temp_path = Path(f.name)
        try:
            expected_digest = "de911f001d77f932396d3ad554003c9d2068a1f6dded1be5b34cefbd8b39c5db"
            self.assertEqual(calculate_sha256(temp_path), expected_digest)
            self.assertTrue(verify_file_sha256(temp_path, expected_digest))
            self.assertFalse(verify_file_sha256(temp_path, "0" * 64))
        finally:
            temp_path.unlink(missing_ok=True)

    def test_safety_hf_revision_required_and_hex40(self):
        """SAFETY_HF_REPO overrides require an immutable 40-hex revision."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            comfy_dir = Path(tmp_dir) / "comfy"

            for bad in (None, "main", "master", "v1.0", "1a2b3c", "0123456789abcdef"):
                with self.assertRaises(ValueError):
                    ensure_model_assets(
                        model_dir=model_dir, comfy_dir=comfy_dir,
                        safety_hf_repo="mitchins/comfyui-junior-safety",
                        safety_hf_revision=bad, dry_run=True, stack="blackwell_nvfp4")

    def test_offline_safety_snapshot_uses_pinned_revision(self):
        """A valid override revision flows into the snapshot download call."""
        valid_sha = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            comfy_dir = Path(tmp_dir) / "comfy"
            (model_dir / "diffusion_models").mkdir(parents=True)
            (model_dir / "text_encoders").mkdir(parents=True)
            (model_dir / "vae").mkdir(parents=True)
            (model_dir / "safety").mkdir(parents=True)
            (model_dir / "diffusion_models" / "flux-2-klein-4b-nvfp4.safetensors").write_bytes(b"dummy")
            (model_dir / "text_encoders" / "qwen_3_4b_fp4_flux2.safetensors").write_bytes(b"dummy")
            (model_dir / "vae" / "flux2-vae.safetensors").write_bytes(b"dummy")
            (model_dir / "safety" / "lid.176.bin").write_bytes(b"dummy")

            with unittest.mock.patch("huggingface_hub.snapshot_download") as mock_snap, \
                 unittest.mock.patch("comfyui_junior.model_assets.verify_directory_assets", return_value=True):
                res = ensure_model_assets(
                    model_dir=model_dir, comfy_dir=comfy_dir,
                    safety_hf_repo="mitchins/comfyui-junior-safety",
                    safety_hf_revision=valid_sha, dry_run=False,
                    stack="blackwell_nvfp4")
                mock_snap.assert_called_once()
                kwargs = mock_snap.call_args.kwargs
                self.assertEqual(kwargs.get("repo_id"), "mitchins/comfyui-junior-safety")
                self.assertEqual(kwargs.get("revision"), valid_sha)
                self.assertTrue(res.get("junior-safety-v29db"))


class TestComfyNodeResolution(unittest.TestCase):
    """Role-based node discovery in the ComfyUI client."""

    def test_roles_resolved_by_class_type(self):
        """Required roles are found regardless of node ids."""
        client = ComfyClient.__new__(ComfyClient)
        wf = {
            "11": {"class_type": "CLIPTextEncode", "inputs": {}},
            "22": {"class_type": "EmptyLatentImage", "inputs": {}},
            "33": {"class_type": "KSampler", "inputs": {}},
        }
        roles = ComfyClient._index_roles(wf)
        self.assertEqual(roles["text"], "11")
        self.assertEqual(roles["latent"], "22")
        self.assertEqual(roles["sampler"], "33")

    def test_missing_role_raises(self):
        """A workflow without a sampler is rejected at load."""
        with self.assertRaises(ValueError):
            ComfyClient._index_roles({"1": {"class_type": "CLIPTextEncode", "inputs": {}}})

    def test_metadata_keys_ignored(self):
        """Non-dict entries (e.g. comments) do not break discovery."""
        roles = ComfyClient._index_roles({
            "comment": "hello",
            "1": {"class_type": "CLIPTextEncode", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {}},
        })
        self.assertEqual(roles["text"], "1")

    def test_text_role_follows_sampler_positive_link(self):
        """With distinct encoders, the prompt goes to the positive one."""
        wf = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {}},
            "4": {"class_type": "KSampler", "inputs": {"positive": ["2", 0], "negative": ["1", 0]}},
        }
        roles = ComfyClient._index_roles(wf)
        self.assertEqual(roles["text"], "2")

    def test_plaintext_remote_workflow_rejected(self):
        """http:// workflow sources are refused; https:// is the only remote scheme."""
        valid = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
            "2": {"class_type": "EmptyLatentImage", "inputs": {}},
            "3": {"class_type": "KSampler", "inputs": {}},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(valid, f)
            path = f.name
        try:
            ComfyClient(base_url="http://127.0.0.1:8188", workflow_path=path)  # local path ok
        finally:
            Path(path).unlink(missing_ok=True)
        with self.assertRaises(ValueError):
            ComfyClient(base_url="http://127.0.0.1:8188", workflow_path="http://example.com/wf.json")
        with self.assertRaises(FileNotFoundError):
            ComfyClient(base_url="http://127.0.0.1:8188", workflow_path="ftp://example.com/wf.json")

    def test_workflow_redirects_cannot_downgrade_to_http(self):
        """HTTPS workflow URLs may not be redirected to plaintext targets."""
        from urllib.request import Request

        from comfyui_junior.comfy import _HTTPSOnlyRedirectHandler

        handler = _HTTPSOnlyRedirectHandler()
        req = Request("https://example.com/workflow.json")
        headers = {"Location": "http://attacker.example/workflow.json"}
        with self.assertRaises(ValueError) as ctx:
            handler.redirect_request(req, None, 302, "Found", headers, headers["Location"])
        self.assertIn("not HTTPS", str(ctx.exception))

        # HTTPS-to-HTTPS redirects are delegated to the normal handler flow
        # (returns a request object or None, but never raises for scheme).
        https_target = "https://cdn.example.com/workflow.json"
        result = handler.redirect_request(req, None, 302, "Found",
                                          {"Location": https_target}, https_target)
        self.assertNotIsInstance(result, ValueError)


class TestWorkflowValidation(unittest.TestCase):
    """Both packaged workflows contain the required tiled-decode graph."""

    def _validate(self, name):
        path = WORKFLOWS / name
        self.assertTrue(path.exists(), f"Workflow JSON missing at {path}")
        wf = json.loads(path.read_text(encoding="utf-8"))
        node_types = {n["class_type"] for n in wf.values() if isinstance(n, dict)}
        for required in ("UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
                         "KSampler", "VAEDecodeTiled", "SaveImage"):
            self.assertIn(required, node_types)
        tiled = next(n for n in wf.values()
                     if isinstance(n, dict) and n["class_type"] == "VAEDecodeTiled")
        self.assertEqual(tiled["inputs"]["tile_size"], 512)
        self.assertEqual(tiled["inputs"]["overlap"], 64)

    def test_blackwell_workflow(self):
        """The NVFP4 (Blackwell) workflow is intact."""
        self._validate("flux2_klein_4b.json")

    def test_ampere_workflow(self):
        """The fp8 (Ampere) workflow is intact."""
        self._validate("flux2_klein_fp8_ampere.json")
        wf = json.loads((WORKFLOWS / "flux2_klein_fp8_ampere.json").read_text(encoding="utf-8"))
        unet = next(n for n in wf.values() if isinstance(n, dict) and n["class_type"] == "UNETLoader")
        self.assertEqual(unet["inputs"]["unet_name"], "flux-2-klein-4b-fp8.safetensors")
        self.assertNotIn("nvfp4", json.dumps(wf))


class TestClassifierManifestValidation(unittest.TestCase):
    """The classifier fails closed on manifest/digest mismatches."""

    def _init_classifier(self, model_path, manifest):
        with unittest.mock.patch("comfyui_junior.model_assets.load_manifest", return_value=manifest):
            from comfyui_junior.classifier import JuniorSafetyClassifier
            return JuniorSafetyClassifier(model_dir=str(model_path), device="cpu")

    def test_missing_manifest_record_fails(self):
        """No pinned record means the classifier refuses to construct."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "heads.pt").write_bytes(b"dummy")
            with self.assertRaises(ValueError) as ctx:
                self._init_classifier(Path(tmp_dir), {"models": []})
            self.assertIn("exactly one 'junior-safety-v29db'", str(ctx.exception))

    def test_empty_files_declaration_fails(self):
        """A record declaring no files must not pass verification vacuously."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "heads.pt").write_bytes(b"dummy")
            bad = {"models": [{"id": "junior-safety-v29db",
                               "files": [], "file_digests": {}}]}
            with self.assertRaises(ValueError) as ctx:
                self._init_classifier(Path(tmp_dir), bad)
            self.assertIn("declares no expected files", str(ctx.exception))

    def test_incomplete_serving_files_declaration_fails(self):
        """Records missing model.safetensors/heads.pt/tokenizer files are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "heads.pt").write_bytes(b"dummy")
            bad = {"models": [{"id": "junior-safety-v29db",
                               "files": ["heads.pt"],
                               "file_digests": {"heads.pt": "0" * 64}}]}
            with self.assertRaises(ValueError) as ctx:
                self._init_classifier(Path(tmp_dir), bad)
            self.assertIn("missing required serving files", str(ctx.exception))

    def test_missing_digest_fails(self):
        """A record without a digest for an expected file is rejected."""
        required = ["model.safetensors", "heads.pt", "config.json", "tokenizer.json"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in required:
                (Path(tmp_dir) / name).write_bytes(b"dummy")
            bad = {"models": [{"id": "junior-safety-v29db",
                               "files": required + ["spm.model"],
                               "file_digests": {f: "0" * 64 for f in required}}]}
            with self.assertRaises(ValueError) as ctx:
                self._init_classifier(Path(tmp_dir), bad)
            self.assertIn("missing digest", str(ctx.exception))

    def test_digest_mismatch_fails(self):
        """A file whose digest deviates from the pin is rejected."""
        required = ["model.safetensors", "heads.pt", "config.json", "tokenizer.json"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in required:
                (Path(tmp_dir) / name).write_bytes(b"dummy")
            bad = {"models": [{"id": "junior-safety-v29db",
                               "files": required,
                               "file_digests": {f: "0" * 64 for f in required}}]}
            with self.assertRaises(ValueError) as ctx:
                self._init_classifier(Path(tmp_dir), bad)
            self.assertIn("failed integrity verification", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
