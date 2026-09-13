"""Evaluation-instance mode tests (owner-operated env, reloaded in-process)."""
import base64
import importlib
import os
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient


class TestEvaluationInstanceMode(unittest.TestCase):
    """JUNIOR_EVALUATION_INSTANCE=1 disables gates and the frontend by design."""

    @classmethod
    def setUpClass(cls):
        """Reload config/app with the evaluation env var set."""
        cls._prev = os.environ.get("JUNIOR_EVALUATION_INSTANCE")
        os.environ["JUNIOR_EVALUATION_INSTANCE"] = "1"
        import comfyui_junior.config as config_mod
        import comfyui_junior.app as app_mod
        cls._config_mod, cls._app_mod = config_mod, app_mod
        cls._config = importlib.reload(config_mod)
        cls._app = importlib.reload(app_mod)

    @classmethod
    def tearDownClass(cls):
        """Restore appliance-mode modules and environment."""
        if cls._prev is None:
            os.environ.pop("JUNIOR_EVALUATION_INSTANCE", None)
        else:
            os.environ["JUNIOR_EVALUATION_INSTANCE"] = cls._prev
        importlib.reload(cls._config_mod)
        importlib.reload(cls._app_mod)

    def test_health_marks_evaluation_role(self):
        """/health exposes role=evaluation and the gates-disabled marker."""
        dummy_png = b"\x89PNG\r\n\x1a\n"
        with patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            mock_comfy.return_value.check_health.return_value = True
            with TestClient(self._app.app) as client:
                data = client.get("/health").json()
                self.assertEqual(data["role"], "evaluation")
                self.assertIn("disabled-by-owner", data["gates"])

    def test_frontend_disabled(self):
        """The child-facing studio is replaced by an evaluation notice."""
        with patch("comfyui_junior.app.ComfyClient"):
            with TestClient(self._app.app) as client:
                resp = client.get("/")
                data = resp.json()
                self.assertEqual(data.get("role"), "evaluation")
                self.assertIn("Evaluation instance", data.get("message", ""))

    def test_unsafe_prompt_generates_by_design(self):
        """On evaluation instances even BLOCK-class prompts reach the backend."""
        dummy_png = b"\x89PNG\r\n\x1a\n"
        with patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            mock_comfy.return_value.generate_image.return_value = (dummy_png, 0.5)
            with TestClient(self._app.app) as client:
                resp = client.post("/v1/images/generations", json={"prompt": "explicit prohibited prompt"})
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(base64.b64decode(resp.json()["data"][0]["b64_json"]), dummy_png)
                mock_comfy.return_value.generate_image.assert_called_once()


if __name__ == "__main__":
    unittest.main()
