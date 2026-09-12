"""ComfyUI backend client: baked workflow + tightly-scoped overrides.

The client resolves nodes in the packaged workflow by ``class_type`` rather
than by literal node id, so re-exported workflows with different ids keep
working. Overrides are limited to the prompt text, latent size, seed and an
optional step count (quality presets); nothing else in the workflow can be
influenced by a request.
"""
import copy
import json
import logging
import random
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("comfyui_junior.comfy")

# Node roles this client is allowed to override, discovered by class_type.
_OVERRIDE_ROLES = {
    "CLIPTextEncode": "text",
    "EmptyLatentImage": "latent",
    "EmptySD3LatentImage": "latent",
    "KSampler": "sampler",
    "SaveImage": "save",
}


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses any downgrade away from HTTPS.

    urllib follows redirects by default, so an accepted ``https://`` workflow
    URL could otherwise be answered with a 30x to a plaintext ``http://``
    source. Every redirect target must itself be HTTPS; anything else raises.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https":
            raise ValueError(
                f"workflow redirect refused: {newurl!r} is not HTTPS (downgrade blocked)"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _https_only_opener() -> urllib.request.OpenerDirector:
    """Build a URL opener that follows only HTTPS-to-HTTPS redirects."""
    return urllib.request.build_opener(_HTTPSOnlyRedirectHandler)


class ComfyClient:
    """Submits the packaged workflow to ComfyUI and retrieves the PNG bytes."""

    def __init__(self, base_url: str, workflow_path: str, quality_steps: Optional[Dict[str, int]] = None):
        """Load and validate a baked workflow template.

        Args:
            base_url: ComfyUI base URL (e.g. ``http://127.0.0.1:8188``).
            workflow_path: path to the API-format workflow JSON.
            quality_steps: optional mapping of quality name to sampler steps.

        Raises:
            FileNotFoundError: when the workflow file does not exist.
            ValueError: when the workflow is missing a required node role.
        """
        self.base_url = base_url.rstrip("/")
        self.quality_steps = quality_steps or {}
        if workflow_path and (workflow_path.startswith("https://") or workflow_path.startswith("http://")):
            # Remote workflow sources are an operator-supplied override:
            # plaintext http:// is rejected (tampering vector) and only
            # https:// is accepted. Local paths remain the default.
            if not workflow_path.startswith("https://"):
                raise ValueError("remote workflow sources must use https:// (plaintext http is not allowed)")
            with urllib.request.urlopen(workflow_path, timeout=10.0,
                                        opener=_https_only_opener()) as resp:
                self.workflow_template: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        else:
            import os
            if not workflow_path or not os.path.exists(workflow_path):
                raise FileNotFoundError(f"Workflow template not found: {workflow_path}")
            with open(workflow_path, "r") as f:
                self.workflow_template = json.load(f)
        self._roles = self._index_roles(self.workflow_template)
        logger.info("Loaded Comfy workflow %s (roles=%s)", workflow_path, self._roles)

    @staticmethod
    def _index_roles(workflow: Dict[str, Any]) -> Dict[str, str]:
        """Map node roles to ids by scanning for known class types.

        Args:
            workflow: API-format workflow graph.

        Returns:
            Mapping of role name (``text``/``latent``/``sampler``/``save``) to
            node id.

        Raises:
            ValueError: when a required role is missing.
        """
        roles: Dict[str, str] = {}
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue  # allow metadata keys (e.g. "comment") in the JSON
            ct = node.get("class_type", "")
            if ct in _OVERRIDE_ROLES and _OVERRIDE_ROLES[ct] not in roles:
                roles[_OVERRIDE_ROLES[ct]] = nid
        # Prefer the text encoder actually wired to the sampler's positive
        # input: with distinct positive/negative encoders, writing the prompt
        # into whichever CLIPTextEncode appears first would be wrong.
        sampler_id = roles.get("sampler")
        if sampler_id:
            positive = workflow[sampler_id].get("inputs", {}).get("positive")
            if isinstance(positive, list) and positive and str(positive[0]) in workflow:
                linked = workflow[str(positive[0])]
                if isinstance(linked, dict) and linked.get("class_type") == "CLIPTextEncode":
                    roles["text"] = str(positive[0])
        missing = [r for r in ("text", "latent", "sampler") if r not in roles]
        if missing:
            raise ValueError(f"workflow missing required node roles: {missing}")
        return roles

    def check_health(self) -> bool:
        """Return True when the ComfyUI backend answers ``/system_stats``."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("ComfyUI health check failed: %s", e)
            return False

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> Tuple[bytes, float]:
        """Render one image through the packaged workflow.

        Args:
            prompt: fully style-expanded prompt text to encode.
            width: latent width in pixels (multiple of 16, enforced upstream).
            height: latent height in pixels.
            seed: optional deterministic seed; random when ``None``.
            steps: optional sampler step override (quality presets).

        Returns:
            Tuple of ``(png_bytes, wall_latency_seconds)``.

        Raises:
            RuntimeError: on submission errors or execution errors.
            TimeoutError: when ComfyUI does not finish within the poll window.
        """
        t_start = time.perf_counter()
        workflow = copy.deepcopy(self.workflow_template)

        if seed is None:
            seed = random.randint(1, 2**31 - 1)

        text_node = workflow[self._roles["text"]]
        if "text" in text_node["inputs"]:
            text_node["inputs"]["text"] = prompt
        else:
            raise KeyError("text node has no 'text' input")

        latent_node = workflow[self._roles["latent"]]
        latent_node["inputs"]["width"] = int(width)
        latent_node["inputs"]["height"] = int(height)
        latent_node["inputs"]["batch_size"] = 1

        sampler_node = workflow[self._roles["sampler"]]
        sampler_node["inputs"]["seed"] = int(seed)
        if steps is not None:
            sampler_node["inputs"]["steps"] = int(steps)

        img_bytes = self._submit_and_fetch(workflow)
        return img_bytes, time.perf_counter() - t_start

    def _submit(self, workflow: Dict[str, Any]) -> str:
        """Submit a workflow and return the ComfyUI prompt id."""
        payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {result}")
        return prompt_id

    def _wait_outputs(self, prompt_id: str, max_wait_seconds: float = 300.0) -> Dict[str, Any]:
        """Poll history until the prompt completes; return its outputs.

        Args:
            prompt_id: id returned by :meth:`_submit`.
            max_wait_seconds: polling budget.

        Returns:
            The ``outputs`` mapping from the finished history entry.

        Raises:
            RuntimeError: when ComfyUI reports an execution error.
            TimeoutError: when the polling budget is exhausted.
        """
        start = time.time()
        while time.time() - start < max_wait_seconds:
            try:
                with urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}", timeout=5.0) as resp:
                    history = json.loads(resp.read().decode("utf-8"))
            except Exception:
                time.sleep(0.1)
                continue
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution error: {status}")
                outputs = entry.get("outputs", {})
                if outputs:
                    return outputs
            time.sleep(0.1)
        raise TimeoutError(f"ComfyUI execution timed out after {max_wait_seconds}s for {prompt_id}")

    def _fetch_image(self, outputs: Dict[str, Any]) -> Tuple[bytes, Dict[str, str]]:
        """Fetch the first image bytes referenced by workflow outputs."""
        for node_output in outputs.values():
            if "images" in node_output and node_output["images"]:
                info = node_output["images"][0]
                params = urllib.parse.urlencode({
                    "filename": info.get("filename", ""),
                    "subfolder": info.get("subfolder", ""),
                    "type": info.get("type", "output"),
                })
                with urllib.request.urlopen(f"{self.base_url}/view?{params}", timeout=30.0) as view_resp:
                    return view_resp.read(), info
        raise RuntimeError("no image in ComfyUI outputs")

    def _submit_and_fetch(self, workflow: Dict[str, Any]) -> bytes:
        """Submit a workflow and block until its image bytes are retrieved."""
        prompt_id = self._submit(workflow)
        outputs = self._wait_outputs(prompt_id)
        img_bytes, _ = self._fetch_image(outputs)
        logger.info("ComfyUI prompt %s completed (%d bytes)", prompt_id, len(img_bytes))
        return img_bytes
