import os
import json
import time
import uuid
import random
import logging
import copy
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("comfyui_junior.comfy")

def find_unique_node(workflow: Dict[str, Any], class_type: str, fallback_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Locates a node by class_type if uniquely identifiable in the workflow template.
    Explicitly detects and raises on missing or ambiguous duplicate matches.
    """
    matching = [(nid, n) for nid, n in workflow.items() if isinstance(n, dict) and n.get("class_type") == class_type]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise ValueError(f"Ambiguous workflow template: found {len(matching)} nodes of class_type '{class_type}'")
    if fallback_id and fallback_id in workflow and isinstance(workflow[fallback_id], dict):
        if workflow[fallback_id].get("class_type") == class_type:
            return fallback_id, workflow[fallback_id]
    raise KeyError(f"Required node with class_type '{class_type}' not found in workflow template")

class ComfyClient:
    def __init__(self, base_url: str, workflow_path: Path):
        self.base_url = base_url.rstrip("/")
        self.workflow_path = Path(workflow_path)
        
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"Workflow template not found: {self.workflow_path}")
            
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            self.workflow_template: Dict[str, Any] = json.load(f)
            
        # Verify that expected nodes exist uniquely in template
        find_unique_node(self.workflow_template, "CLIPTextEncode", "4")
        find_unique_node(self.workflow_template, "EmptyLatentImage", "5")
        find_unique_node(self.workflow_template, "KSampler", "6")
        
        logger.info("Loaded baked Comfy workflow template from %s", self.workflow_path)

    def check_health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug("ComfyUI health check failed: %s", e)
            return False

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, seed: Optional[int] = None) -> Tuple[bytes, float]:
        """
        Clones baked workflow, applies permitted overrides (prompt, width, height, seed),
        submits to ComfyUI, waits for completion, and returns the raw PNG image bytes and latency.
        """
        t_start = time.perf_counter()
        workflow = copy.deepcopy(self.workflow_template)
        
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
            
        # Strictly override only authorized nodes
        # 1. CLIPTextEncode (Prompt)
        _, text_node = find_unique_node(workflow, "CLIPTextEncode", "4")
        if "inputs" not in text_node:
            text_node["inputs"] = {}
        text_node["inputs"]["text"] = prompt
            
        # 2. EmptyLatentImage (Dimensions)
        _, latent_node = find_unique_node(workflow, "EmptyLatentImage", "5")
        if "inputs" not in latent_node:
            latent_node["inputs"] = {}
        latent_node["inputs"]["width"] = int(width)
        latent_node["inputs"]["height"] = int(height)
        latent_node["inputs"]["batch_size"] = 1
            
        # 3. KSampler (Seed)
        _, sampler_node = find_unique_node(workflow, "KSampler", "6")
        if "inputs" not in sampler_node:
            sampler_node["inputs"] = {}
        sampler_node["inputs"]["seed"] = int(seed)
            
        client_id = str(uuid.uuid4())
        payload = {
            "prompt": workflow,
            "client_id": client_id
        }
        
        # Submit Prompt
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error("Failed to submit prompt to ComfyUI: %s", e)
            raise RuntimeError(f"ComfyUI submission error: {e}")
            
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {result}")
            
        logger.info("Submitted prompt %s to ComfyUI (seed=%d, size=%dx%d)", prompt_id, seed, width, height)
        
        # Poll for Completion
        max_wait_seconds = 120.0
        poll_interval = 0.1
        start_poll = time.time()
        
        while time.time() - start_poll < max_wait_seconds:
            history_url = f"{self.base_url}/history/{prompt_id}"
            try:
                with urllib.request.urlopen(history_url, timeout=5.0) as resp:
                    history = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                logger.debug("Polling error for prompt %s: %s", prompt_id, e)
                time.sleep(poll_interval)
                continue
                
            if prompt_id in history:
                prompt_data = history[prompt_id]
                
                # Check for execution failure immediately
                status_info = prompt_data.get("status", {})
                if status_info:
                    status_str = status_info.get("status_str")
                    completed = status_info.get("completed", False)
                    if status_str == "error" or (not completed and status_info.get("messages")):
                        messages = status_info.get("messages", [])
                        raise RuntimeError(f"ComfyUI execution failed for prompt {prompt_id}: {messages}")

                outputs = prompt_data.get("outputs", {})
                # Check SaveImage output
                for node_id, node_output in outputs.items():
                    if "images" in node_output and len(node_output["images"]) > 0:
                        img_info = node_output["images"][0]
                        filename = img_info.get("filename")
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")
                        
                        # Retrieve Generated Image
                        view_params = urllib.parse.urlencode({
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type
                        })
                        view_url = f"{self.base_url}/view?{view_params}"
                        with urllib.request.urlopen(view_url, timeout=10.0) as view_resp:
                            img_bytes = view_resp.read()
                            
                        latency_s = time.perf_counter() - t_start
                        logger.info("Completed ComfyUI prompt %s in %.3fs (%d bytes)", prompt_id, latency_s, len(img_bytes))
                        return img_bytes, latency_s
                        
            time.sleep(poll_interval)
            
        raise TimeoutError(f"ComfyUI execution timed out after {max_wait_seconds}s for prompt {prompt_id}")
