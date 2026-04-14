"""RunPod Serverless handler for GLM-OCR (zai-org/GLM-OCR).

Accepts base64-encoded images, returns OCR text.
Uses transformers pipeline directly (0.9B model fits easily in 24GB VRAM).

Cold start behavior:
- All imports and model loading happen at module level (before start())
- RUNPOD_INIT_TIMEOUT env var (default 600s) gives the worker enough time
  to download and load the ~2GB model before RunPod's health check kills it
- runpod.serverless.start() blocks as the main event loop
"""

import sys
import base64
import io
import os
import time
import traceback

print(f"[glm-ocr] Python {sys.version}", flush=True)

# --- Heavy imports with diagnostic logging ---
try:
    print("[glm-ocr] Importing torch...", flush=True)
    import torch
    import torch.nn.functional as F
    print(f"[glm-ocr] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[glm-ocr] GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"[glm-ocr] FATAL: torch import failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[glm-ocr] Importing transformers...", flush=True)
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor
    print(f"[glm-ocr] transformers {transformers.__version__}", flush=True)
except Exception as e:
    print(f"[glm-ocr] FATAL: transformers import failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

from PIL import Image

print("[glm-ocr] Importing runpod...", flush=True)
import runpod

# --- Model loading at module level (standard RunPod pattern) ---
# This runs before runpod.serverless.start(). Set RUNPOD_INIT_TIMEOUT=600
# in endpoint env vars so RunPod waits for this to complete.

MODEL_NAME = os.environ.get("MODEL_NAME", "zai-org/GLM-OCR")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"[glm-ocr] Loading {MODEL_NAME}@{MODEL_REVISION}", flush=True)
print(f"[glm-ocr] HF_HOME: {HF_HOME}", flush=True)

processor = None
model = None
tokenizer = None
load_error = None

try:
    t0 = time.time()

    print("[glm-ocr] Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )

    print("[glm-ocr] Loading model weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    # --- Monkey-patch Conv3d → F.linear for patch embedding ---
    # GLM-OCR's vision encoder produces ~22k patches, each dispatching a
    # separate CUDA Conv3d kernel. This is the known perf bottleneck.
    try:
        base_model = model.model if hasattr(model, "model") else model
        if hasattr(base_model, "visual") and hasattr(base_model.visual, "patch_embed"):
            patch_embed = base_model.visual.patch_embed
            proj = patch_embed.proj
            in_features = (
                patch_embed.in_channels
                * patch_embed.temporal_patch_size
                * patch_embed.patch_size ** 2
            )
            embed_dim = patch_embed.embed_dim
            weight = proj.weight
            bias = proj.bias

            def _fast_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                target_dtype = weight.dtype
                hidden_states = hidden_states.reshape(-1, in_features).to(dtype=target_dtype)
                return F.linear(hidden_states, weight.reshape(embed_dim, -1), bias)

            patch_embed.forward = _fast_forward
            print("[glm-ocr] Applied Conv3d→linear monkey patch", flush=True)
    except Exception as mp_err:
        print(f"[glm-ocr] WARN: Monkey patch failed (non-fatal): {mp_err}", flush=True)

    elapsed = time.time() - t0
    print(f"[glm-ocr] Model ready in {elapsed:.1f}s. Device map: {model.hf_device_map}", flush=True)

except Exception as e:
    load_error = e
    print(f"[glm-ocr] ERROR loading model: {e}", flush=True)
    traceback.print_exc()


# --- Helper functions ---

def decode_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    img_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def ocr_single_image(image: Image.Image, prompt: str = "OCR:") -> str:
    """Run OCR on a single image and return extracted text."""
    t0 = time.time()
    model_device = next(model.parameters()).device
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=4096, do_sample=False
        )

    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    output_ids = generated_ids[:, input_len:]
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"[glm-ocr] Generated {len(output_ids[0])} tokens in {elapsed:.1f}s", flush=True)
    return text.strip()


# --- RunPod handler ---

def handler(event):
    """RunPod serverless handler.

    Input formats:
      1. Single image:  {"input": {"image": "<base64>", "prompt": "OCR:"}}
      2. Multiple images: {"input": {"images": ["<base64>", ...], "prompt": "OCR:"}}

    Returns:
      Single:   {"response": "extracted text"}
      Multiple: {"results": [{"index": 0, "response": "..."}, ...]}
      Error:    {"error": "message"}
    """
    if load_error is not None:
        return {"error": f"Model failed to load: {load_error}"}

    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "OCR:")

        if "image" in job_input:
            image = decode_image(job_input["image"])
            text = ocr_single_image(image, prompt)
            return {"response": text}

        if "images" in job_input:
            results = []
            for i, img_data in enumerate(job_input["images"]):
                try:
                    image = decode_image(img_data)
                    text = ocr_single_image(image, prompt)
                    results.append({"index": i, "response": text})
                except Exception as img_err:
                    results.append({"index": i, "response": "", "error": str(img_err)})
            return {"results": results}

        return {"error": "No 'image' or 'images' field in input"}

    except Exception as e:
        torch.cuda.empty_cache()
        return {"error": str(e)}


# --- RunPod entrypoint ---
# start() blocks — it IS the main event loop. Everything above runs first.
# RUNPOD_INIT_TIMEOUT env var controls how long RunPod waits before this point.
print("[glm-ocr] Calling runpod.serverless.start()...", flush=True)
runpod.serverless.start({"handler": handler})
