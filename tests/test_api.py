"""API tests: validation, gate wiring (classifier mocked), generation flow."""
import base64
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from comfyui_junior.app import app as _APP
from comfyui_junior.classifier import ClassificationResult, DIMS


def _classifier_result(decision="PASS"):
    """Build a canned classifier result for the stub gate."""
    level = 0 if decision == "PASS" else 3
    return ClassificationResult(
        decision=decision,
        levels={d: level for d in DIMS},
        probabilities={d: [0.9] * w for d, w in zip(DIMS, [3, 2, 2, 2, 2, 2])},
        reasons=["sexual level 3 >= 2"] if decision == "BLOCK" else [],
    )


class _StubClassifier:
    """Deterministic stand-in for the FP16 v29db classifier.

    The decision is a class attribute (``outcome``) so tests can retarget it
    after the lifespan has already constructed the gate.
    """

    dtype = __import__("torch").float16
    device = __import__("torch").device("cuda:0")
    outcome = "PASS"

    def __init__(self, *args, **kwargs):
        pass

    def classify(self, prompt):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return _classifier_result(self.outcome)


def _make_fake_lid(label="en", prob=0.95):
    """Build a fastText stand-in with the native-binding shape (model.f)."""
    class _Binding:
        def predict(self, *args, **kwargs):
            return [(prob, f"__label__{label}")]
    class _Model:
        def __init__(self):
            self.f = _Binding()
    return _Model()


class TestAPIEndpoints(unittest.TestCase):
    """Endpoint behaviour with the classifier stubbed and backend mocked."""

    def setUp(self):
        _StubClassifier.outcome = "PASS"
        self._lid_patch = patch("comfyui_junior.langgate._model", _make_fake_lid())
        self._lid_patch.start()
        self.addCleanup(self._lid_patch.stop)

    def _client(self):
        return TestClient(_APP)

    def test_health_endpoint(self):
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            mock_comfy.return_value.check_health.return_value = True
            with self._client() as client:
                resp = client.get("/health")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["role"], "appliance")
                self.assertIn("v29db_fp16", data["gates"])
                self.assertEqual(data["public_model"], "flux2-klein-safe")

    def test_models_endpoint(self):
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient"):
            with self._client() as client:
                resp = client.get("/v1/models")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["data"][0]["id"], "flux2-klein-safe")

    def test_root_index_serving(self):
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient"):
            with self._client() as client:
                resp = client.get("/")
                self.assertEqual(resp.status_code, 200)
                self.assertIn("Imagine", resp.text)

    def test_static_assets_serving(self):
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient"):
            with self._client() as client:
                self.assertEqual(client.get("/static/app.css").status_code, 200)
                self.assertEqual(client.get("/static/app.js").status_code, 200)
                self.assertEqual(client.get("/static/vendor/alpine.min.js").status_code, 200)

    def test_request_validation(self):
        """Model, size, quality, response_format and style are validated."""
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient"):
            with self._client() as client:
                cases = [
                    ({"prompt": "a puppy", "model": "dall-e-3"}, "model_not_found"),
                    ({"prompt": "a puppy", "size": "invalid"}, "invalid_size"),
                    ({"prompt": "a puppy", "size": "1000x1000"}, "invalid_size"),   # not %16
                    ({"prompt": "a puppy", "size": "2048x2048"}, "invalid_size"),   # > 1344
                    ({"prompt": "a puppy", "response_format": "url"}, "invalid_response_format"),
                    ({"prompt": "a puppy", "quality": "ultra"}, "invalid_quality"),
                    ({"prompt": "a puppy", "style": "hyperreal"}, "invalid_style"),
                ]
                for body, code in cases:
                    with self.subTest(code=code):
                        resp = client.post("/v1/images/generations", json=body)
                        self.assertEqual(resp.status_code, 400, body)
                        self.assertEqual(resp.json()["error"]["code"], code)

    def test_null_size_rejected_by_schema(self):
        """An explicit null size fails schema validation instead of crashing."""
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient"):
            with self._client() as client:
                resp = client.post("/v1/images/generations",
                                   json={"prompt": "a puppy", "size": None})
                self.assertEqual(resp.status_code, 422)

    def test_backend_error_is_generic_for_clients(self):
        """Backend failures return a fixed message without internal details."""
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            mock_comfy.return_value.generate_image.side_effect = RuntimeError(
                "secret internal path /app/ComfyUI leak")
            with self._client() as client:
                resp = client.post("/v1/images/generations", json={"prompt": "a puppy"})
                self.assertEqual(resp.status_code, 502)
                message = resp.json()["error"]["message"]
                self.assertNotIn("secret", message)
                self.assertNotIn("ComfyUI", message)
                self.assertIn("couldn't be made", message)

    def test_generation_pass_flow_with_style_template(self):
        """A passing prompt generates and reports the expanded render prompt."""
        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            mock_comfy.return_value.generate_image.return_value = (dummy_png, 1.25)
            with self._client() as client:
                resp = client.post("/v1/images/generations", json={
                    "prompt": "a cute penguin",
                    "style": "colouring_sheet",
                    "size": "1024x1024",
                })
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(base64.b64decode(data["data"][0]["b64_json"]), dummy_png)
                self.assertIn("line art", data["data"][0]["revised_prompt"])
                self.assertEqual(data["meta"]["style_template"], "colouring_sheet")
                kwargs = mock_comfy.return_value.generate_image.call_args.kwargs
                self.assertEqual(kwargs["steps"], 4)  # normal quality for the stack

    def test_malformed_prompt_rejected_zero_comfy_calls(self):
        """Obfuscated input is rejected at stage 1 without backend submission."""
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            with self._client() as client:
                resp = client.post("/v1/images/generations", json={"prompt": "n4k3d p3opl3"})
                self.assertEqual(resp.status_code, 400)
                self.assertEqual(resp.json()["error"]["code"], "prompt_format_invalid")
                mock_comfy.return_value.generate_image.assert_not_called()

    def test_safety_block_zero_comfy_calls(self):
        """A classifier BLOCK yields content_policy_violation with no backend call."""
        _StubClassifier.outcome = "BLOCK"
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            with self._client() as client:
                resp = client.post("/v1/images/generations", json={"prompt": "explicit prohibited prompt"})
                self.assertEqual(resp.status_code, 400)
                data = resp.json()
                self.assertEqual(data["error"]["code"], "content_policy_violation")
                self.assertEqual(data["safety_failure_code"], "content_policy_violation")
                mock_comfy.return_value.generate_image.assert_not_called()

    def test_classifier_exception_fails_closed(self):
        """A crashing classifier fails closed and never reaches the backend."""
        _StubClassifier.outcome = RuntimeError("gpu memory exception")
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            with self._client() as client:
                resp = client.post("/v1/images/generations", json={"prompt": "test prompt"})
                self.assertEqual(resp.status_code, 400)
                self.assertEqual(resp.json()["error"]["code"], "content_policy_violation")
                mock_comfy.return_value.generate_image.assert_not_called()

    def test_bypass_attempts_have_no_effect(self):
        """Request-level safety=false / bypass / headers cannot weaken the gate."""
        _StubClassifier.outcome = "BLOCK"
        with patch("comfyui_junior.app.JuniorSafetyClassifier", _StubClassifier), \
             patch("comfyui_junior.app.ComfyClient") as mock_comfy:
            with self._client() as client:
                attempts = [
                    ("/v1/images/generations?safety=false", {"prompt": "unsafe", "safety": False}),
                    ("/v1/images/generations?bypass=true", {"prompt": "unsafe", "unrestricted": True}),
                    ("/v1/images/generations", {"prompt": "unsafe"}),
                ]
                for url, body in attempts:
                    with self.subTest(url=url):
                        resp = client.post(url, json=body,
                                           headers={"X-Disable-Safety": "1", "X-Bypass-Safety": "true"})
                        self.assertEqual(resp.status_code, 400)
                        self.assertEqual(resp.json()["error"]["code"], "content_policy_violation")
                mock_comfy.return_value.generate_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
