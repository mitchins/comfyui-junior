import json
import base64
import unittest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient

from comfyui_junior.app import app
from comfyui_junior.config import settings
from comfyui_junior.safety import SafetyResult

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        settings.SAFETY_ENABLED = False

    def test_health_endpoint(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("status", data)
            self.assertEqual(data["public_model"], "flux2-klein-4b-safe")

    def test_models_endpoint(self):
        with TestClient(app) as client:
            resp = client.get("/v1/models")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["object"], "list")
            self.assertEqual(data["data"][0]["id"], "flux2-klein-4b-safe")

    def test_root_index_serving(self):
        with TestClient(app) as client:
            resp = client.get("/")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Imagine", resp.text)
            self.assertIn("FLUX.2 Safe", resp.text)

    def test_static_assets_serving(self):
        with TestClient(app) as client:
            resp_css = client.get("/static/app.css")
            self.assertEqual(resp_css.status_code, 200)
            self.assertIn("--bg-app", resp_css.text)

            resp_js = client.get("/static/app.js")
            self.assertEqual(resp_js.status_code, 200)
            self.assertIn("imagineApp", resp_js.text)

            resp_alpine = client.get("/static/vendor/alpine.min.js")
            self.assertEqual(resp_alpine.status_code, 200)

    def test_request_validation_bad_model(self):
        with TestClient(app) as client:
            resp = client.post("/v1/images/generations", json={
                "prompt": "a cute puppy",
                "model": "dall-e-3"
            })
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(resp.json()["error"]["code"], "model_not_found")

    def test_request_validation_bad_size(self):
        with TestClient(app) as client:
            resp = client.post("/v1/images/generations", json={
                "prompt": "a cute puppy",
                "size": "invalid_format"
            })
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(resp.json()["error"]["code"], "invalid_size")

    @patch("comfyui_junior.app.ComfyClient")
    def test_generation_pass_flow(self, mock_comfy_cls):
        mock_instance = mock_comfy_cls.return_value
        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        mock_instance.generate_image.return_value = (dummy_png, 1.25)
        mock_instance.check_health.return_value = True

        with TestClient(app) as client:
            resp = client.post("/v1/images/generations", json={
                "prompt": "a cute penguin in space",
                "size": "1024x1024",
                "response_format": "b64_json"
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("data", data)
            b64_returned = data["data"][0]["b64_json"]
            self.assertEqual(base64.b64decode(b64_returned), dummy_png)
            mock_instance.generate_image.assert_called_once()

    @patch("comfyui_junior.app.ComfyClient")
    @patch("comfyui_junior.app.SafetyFilter")
    def test_safety_block_zero_comfy_calls(self, mock_filter_cls, mock_comfy_cls):
        settings.SAFETY_ENABLED = True
        mock_filter = mock_filter_cls.return_value
        mock_filter.classify.return_value = SafetyResult(
            decision="BLOCK",
            reasons=["sexual_content"],
            scores={"sexual": [1.0, 2.0, 0.5]},
            latency_ms=3.2
        )
        mock_comfy = mock_comfy_cls.return_value
        mock_comfy.check_health.return_value = True

        with TestClient(app) as client:
            resp = client.post("/v1/images/generations", json={
                "prompt": "explicit prohibited prompt",
                "model": "flux2-klein-4b-safe"
            })
            self.assertEqual(resp.status_code, 400)
            data = resp.json()
            self.assertEqual(data["error"]["code"], "content_policy_violation")
            self.assertEqual(data["safety_decision"], "BLOCK")
            mock_comfy.generate_image.assert_not_called()

    @patch("comfyui_junior.app.ComfyClient")
    @patch("comfyui_junior.app.SafetyFilter")
    def test_safety_route_zero_comfy_calls(self, mock_filter_cls, mock_comfy_cls):
        settings.SAFETY_ENABLED = True
        mock_filter = mock_filter_cls.return_value
        mock_filter.classify.return_value = SafetyResult(
            decision="ROUTE",
            reasons=["nudity"],
            scores={"nudity": [0.5, -0.2]},
            latency_ms=2.8
        )
        mock_comfy = mock_comfy_cls.return_value
        mock_comfy.check_health.return_value = True

        with TestClient(app) as client:
            resp = client.post("/v1/images/generations", json={
                "prompt": "swimsuit on beach",
                "model": "flux2-klein-4b-safe"
            })
            self.assertEqual(resp.status_code, 400)
            data = resp.json()
            self.assertEqual(data["error"]["code"], "safety_route_required")
            self.assertEqual(data["safety_decision"], "ROUTE")
            mock_comfy.generate_image.assert_not_called()

    @patch("comfyui_junior.app.ComfyClient")
    @patch("comfyui_junior.app.SafetyFilter")
    def test_safety_exception_fails_closed(self, mock_filter_cls, mock_comfy_cls):
        settings.SAFETY_ENABLED = True
        mock_filter = mock_filter_cls.return_value
        mock_filter.classify.side_effect = RuntimeError("GPU memory exception")
        mock_comfy = mock_comfy_cls.return_value
        mock_comfy.check_health.return_value = True

        with TestClient(app) as client:
            resp = client.post("/v1/images/generations", json={
                "prompt": "test prompt"
            })
            self.assertEqual(resp.status_code, 500)
            data = resp.json()
            self.assertIn("failed", data["error"]["message"].lower())

if __name__ == "__main__":
    unittest.main()
