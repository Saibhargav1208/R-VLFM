# R-VLFM: QwenVL wrapper
# Replaces BLIP2ITM for frontier scoring (Stage 1)
# Adds relational verification (Stage 2) — the new contribution
#
# Drop-in compatible with BLIP2ITMClient:
#   QwenVLClient.cosine(image, text) -> float  (Stage 1, used by value map)
# New methods:
#   QwenVLClient.verify_relation(image, relational_goal) -> float  (Stage 2)
#   QwenVLClient.parse_goal(text) -> dict

import re
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    QWEN_AVAILABLE = True
except ModuleNotFoundError:
    QWEN_AVAILABLE = False
    print("Could not import transformers/Qwen2VL. OK if using client only.")


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

_STAGE1_PROMPT = (
    "Look at this image carefully. "
    "On a scale from 0.0 to 1.0, how likely is it that this scene "
    "leads toward or contains a {object_name}? "
    "Reply with a single decimal number only, e.g. 0.72"
)

_STAGE2_PROMPT = (
    "Look at this image carefully. "
    "Is there {relational_goal} visible in this image? "
    "Consider the spatial relationship precisely. "
    "Reply with a single decimal number from 0.0 (definitely not) "
    "to 1.0 (definitely yes), e.g. 0.85"
)

_PARSE_PROMPT = (
    "Parse this navigation goal into JSON. "
    "Goal: \"{goal_text}\"\n"
    "Return ONLY a JSON object with these exact keys:\n"
    "  object: the target object name (string)\n"
    "  relation: the spatial relation if any, else empty string\n"
    "  is_relational: true if a spatial relation exists, else false\n"
    "Example for 'cup on the kitchen counter':\n"
    "  {{\"object\": \"cup\", \"relation\": \"on the kitchen counter\", \"is_relational\": true}}\n"
    "Example for 'chair':\n"
    "  {{\"object\": \"chair\", \"relation\": \"\", \"is_relational\": false}}\n"
    "Reply with ONLY the JSON, no explanation."
)


def _extract_float(text: str) -> float:
    """Extract first float from model response. Returns 0.0 if none found."""
    text = text.strip()
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return 0.0
    val = float(matches[0])
    # Clamp to [0, 1]
    return float(np.clip(val, 0.0, 1.0))


def _extract_json(text: str) -> dict:
    """Extract JSON dict from model response."""
    import json

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Try to find JSON block inside the text
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Fallback: treat full text as object name
    return {"object": text.strip(), "relation": "", "is_relational": False}


# ──────────────────────────────────────────────
# Server-side model (loaded on GPU)
# ──────────────────────────────────────────────

class QwenVL:
    """
    Qwen2-VL-2B-Instruct wrapper.

    Three capabilities:
      1. score_frontier  — Stage 1: how likely is this image to lead to object?
      2. verify_relation — Stage 2: does the relational goal hold in this image?
      3. parse_goal      — parse natural language goal into structured dict
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[Any] = None,
        max_new_tokens: int = 32,
    ) -> None:
        assert QWEN_AVAILABLE, (
            "transformers not installed. Run: pip install transformers qwen-vl-utils"
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[QwenVL] Loading {model_name} on {device} ...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        self.max_new_tokens = max_new_tokens
        print("[QwenVL] Model loaded.")

    # ── internal helper ──────────────────────────────────────────────────────

    def _run(self, image: np.ndarray, text_prompt: str) -> str:
        """Run Qwen2-VL on a single (image, text) pair. Returns raw text output."""
        pil_img = Image.fromarray(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[chat_text],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return response.strip()

    # ── public API ───────────────────────────────────────────────────────────

    def score_frontier(self, image: np.ndarray, object_name: str) -> float:
        """
        Stage 1 — frontier scoring (replaces BLIP-2 cosine similarity).

        Args:
            image: RGB numpy array (H, W, 3)
            object_name: e.g. "cup", "bed", "chair"

        Returns:
            float in [0, 1] — how likely this scene leads to the object
        """
        prompt = _STAGE1_PROMPT.format(object_name=object_name)
        response = self._run(image, prompt)
        return _extract_float(response)

    def verify_relation(self, image: np.ndarray, relational_goal: str) -> float:
        """
        Stage 2 — relational verification (the new R-VLFM contribution).

        Args:
            image: RGB numpy array (H, W, 3)
            relational_goal: e.g. "a cup on the kitchen counter"

        Returns:
            float in [0, 1] — confidence that the relational goal is satisfied
        """
        prompt = _STAGE2_PROMPT.format(relational_goal=relational_goal)
        response = self._run(image, prompt)
        return _extract_float(response)

    def parse_goal(self, goal_text: str) -> Dict:
        """
        Parse a free-form navigation goal into structured components.

        Args:
            goal_text: e.g. "the red cup on the kitchen counter"

        Returns:
            dict with keys:
                object       (str)  — "cup"
                relation     (str)  — "on the kitchen counter"
                is_relational (bool) — True if relation is non-empty

        Note: uses a blank image for the vision slot (parse is text-only).
        """
        blank = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        prompt = _PARSE_PROMPT.format(goal_text=goal_text)
        response = self._run(blank, prompt)
        return _extract_json(response)

    # ── drop-in for BLIP2ITM ─────────────────────────────────────────────────

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Drop-in replacement for BLIP2ITM.cosine().
        Internally calls score_frontier with the object name extracted from txt.
        """
        return self.score_frontier(image, txt)


# ──────────────────────────────────────────────
# Client (sends requests to QwenVLServer)
# ──────────────────────────────────────────────

class QwenVLClient:
    """
    HTTP client for QwenVLServer.
    Drop-in compatible with BLIP2ITMClient — has .cosine() method.
    Also exposes .verify_relation() and .parse_goal().

    NOTE: matches the single-route pattern used by host_model() in this
    codebase (see blip2itm.py) — one URL, payload differentiates the task
    via an "endpoint" field, server dispatches inside process_payload().
    """

    def __init__(self, port: int = 12190) -> None:
        self.url = f"http://localhost:{port}/qwen_vl"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Drop-in for BLIP2ITMClient.cosine().
        Stage 1: scores how relevant this image is to the object name.
        """
        response = send_request(self.url, endpoint="score", image=image, object_name=txt)
        return float(response["score"])

    def score_frontier(self, image: np.ndarray, object_name: str) -> float:
        """Stage 1 scoring — explicit API."""
        response = send_request(self.url, endpoint="score", image=image, object_name=object_name)
        return float(response["score"])

    def verify_relation(self, image: np.ndarray, relational_goal: str) -> float:
        """Stage 2 relational verification — the new R-VLFM contribution."""
        response = send_request(self.url, endpoint="verify", image=image, relational_goal=relational_goal)
        return float(response["score"])

    def parse_goal(self, goal_text: str) -> Dict:
        """Parse free-form goal text into structured dict."""
        # No image needed for parsing, but send_request requires consistent payload;
        # server side substitutes a blank image internally.
        response = send_request(self.url, endpoint="parse", goal_text=goal_text)
        return response["parsed"]


# ──────────────────────────────────────────────
# Server entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12190)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()

    print(f"[QwenVL Server] Loading model: {args.model}")

    class QwenVLServer(ServerMixin, QwenVL):
        def process_payload(self, payload: dict) -> dict:
            endpoint = payload.get("endpoint", "score")

            if endpoint == "score":
                image = str_to_image(payload["image"])
                object_name = payload["object_name"]
                score = self.score_frontier(image, object_name)
                return {"score": score}

            elif endpoint == "verify":
                image = str_to_image(payload["image"])
                relational_goal = payload["relational_goal"]
                score = self.verify_relation(image, relational_goal)
                return {"score": score}

            elif endpoint == "parse":
                goal_text = payload["goal_text"]
                parsed = self.parse_goal(goal_text)
                return {"parsed": parsed}

            else:
                return {"error": f"Unknown endpoint: {endpoint}"}

    qwen = QwenVLServer(model_name=args.model)
    print(f"[QwenVL Server] Hosting on port {args.port} ...")
    # Single route /qwen_vl, matching the actual host_model() signature
    # (it does not support a routes= kwarg — that was a bug, now fixed)
    host_model(qwen, name="qwen_vl", port=args.port)
