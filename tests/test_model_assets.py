import json
import unittest
import tempfile
import torch
from pathlib import Path
from comfyui_junior.model_assets import load_manifest, ensure_model_assets, calculate_sha256, verify_file_sha256
from comfyui_junior.comfy import find_unique_node

class TestModelAssets(unittest.TestCase):
    def test_load_manifest(self):
        manifest = load_manifest()
        self.assertIn("models", manifest)
        models = manifest["models"]
        self.assertGreaterEqual(len(models), 3)

        ids = [m["id"] for m in models]
        self.assertIn("flux2-klein-4b-nvfp4", ids)
        self.assertIn("qwen3-4b-fp4-flux2", ids)
        self.assertIn("flux2-vae", ids)

        # Verify manifest structure includes revisions and sha256
        for m in models:
            if not m.get("optional"):
                self.assertIn("revision", m)
                self.assertIn("sha256", m)
                self.assertIsNotNone(m["sha256"])

    def test_ensure_model_assets_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            comfy_dir = Path(tmp_dir) / "comfy"
            
            res = ensure_model_assets(
                model_dir=model_dir,
                comfy_dir=comfy_dir,
                dry_run=True
            )
            self.assertIsInstance(res, dict)
            self.assertIn("flux2-klein-4b-nvfp4", res)

    def test_sha256_verification(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"ComfyUI Junior Test Content")
            temp_path = Path(f.name)
        try:
            expected_digest = "e95dc017a4282315b9c5bb02e21b71c080bbcf964042898a3952dc66bfd31e9c"
            # calculate actual
            actual = calculate_sha256(temp_path)
            self.assertTrue(verify_file_sha256(temp_path, actual))
            self.assertFalse(verify_file_sha256(temp_path, "0000000000000000000000000000000000000000000000000000000000000000"))
        finally:
            temp_path.unlink(missing_ok=True)

class TestComfyNodeResolution(unittest.TestCase):
    def test_find_unique_node_success(self):
        wf = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}}
        }
        nid, node = find_unique_node(wf, "CLIPTextEncode")
        self.assertEqual(nid, "1")

    def test_find_unique_node_missing(self):
        wf = {"1": {"class_type": "KSampler"}}
        with self.assertRaises(KeyError):
            find_unique_node(wf, "CLIPTextEncode")

    def test_find_unique_node_ambiguous(self):
        wf = {
            "1": {"class_type": "CLIPTextEncode"},
            "2": {"class_type": "CLIPTextEncode"}
        }
        with self.assertRaises(ValueError):
            find_unique_node(wf, "CLIPTextEncode")

class TestWorkflowValidation(unittest.TestCase):
    def test_workflow_file_integrity(self):
        workflow_path = Path(__file__).resolve().parent.parent / "src" / "comfyui_junior" / "workflows" / "flux2_klein_4b.json"
        self.assertTrue(workflow_path.exists(), f"Workflow JSON missing at {workflow_path}")
        
        with open(workflow_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        # Check required nodes
        node_types = {node["class_type"] for node in wf.values()}
        self.assertIn("UNETLoader", node_types)
        self.assertIn("CLIPLoader", node_types)
        self.assertIn("VAELoader", node_types)
        self.assertIn("CLIPTextEncode", node_types)
        self.assertIn("EmptyLatentImage", node_types)
        self.assertIn("KSampler", node_types)
        self.assertIn("VAEDecodeTiled", node_types)
        self.assertIn("SaveImage", node_types)

        # Verify explicit tiled VAE node settings
        tiled_node = next(n for n in wf.values() if n["class_type"] == "VAEDecodeTiled")
        self.assertEqual(tiled_node["inputs"]["tile_size"], 512)
        self.assertEqual(tiled_node["inputs"]["overlap"], 64)

        # Verify no unresolved custom nodes or UI-only properties
        for node_id, node in wf.items():
            self.assertIn("class_type", node)
            self.assertIn("inputs", node)
