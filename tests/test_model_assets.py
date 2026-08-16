import json
import unittest
from pathlib import Path
from comfyui_junior.model_assets import load_manifest, ensure_model_assets

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

    def test_ensure_model_assets_dry_run(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            comfy_dir = Path(tmp_dir) / "comfy"
            
            res = ensure_model_assets(
                model_dir=model_dir,
                comfy_dir=comfy_dir,
                dry_run=True
            )
            self.assertIsInstance(res, dict)
            # In dry run with empty dir, required models return False
            self.assertIn("flux2-klein-4b-nvfp4", res)

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

if __name__ == "__main__":
    unittest.main()
