"""RunPod Serverless handler for GLM-OCR (zai-org/GLM-OCR).

Accepts base64-encoded images, returns OCR text.
Uses transformers pipeline directly (0.9B model fits easily in 24GB VRAM).

Cold start behavior:
- Worker signals ready immediately (health check passes in ~5s)
- Model downloads in background (~2GB, ~2-3 min on first cold start)
- First request blocks until model is ready; subsequent requests are fast
- Network volume at /runpod-volume/hf-cache (HF_HOME) eliminates download on warm restarts
"""

import base64
import io
import os
import threading
import time
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_NAME = os.environ.get("MODEL_NAME", "zai-org/GLM-OCR")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"[glm-ocr] Worker starting — {MODEL_NAME}@{MODEL_REVISION}")
print(f"[glm-ocr] HF_HOME: {HF_HOME}")

volume_mounted = os.path.isdir("/runpod-volume")
cache_populated = os.path.isdir(os.path.join(HF_HOME, "hub")) and bool(
    os.listdir(os.path.join(HF_HOME, "hub"))
)
if volume_mounted and cache_populated:
    print("[glm-ocr] Network volume cache found — model load will be fast.")
elif volume_mounted:
    print("[glm-ocr] Network volume mounted but cache empty — downloading ~2GB model.")
else:
    print("[glm-ocr] No network volume — downloading model to container disk (~2GB, ~2-3 min).")
    print("[glm-ocr] Attach a network volume at /runpod-volume with pre-cached weights for fast starts.")

# --- Background model loading ---
# Model loads in a daemon thread so runpod.serverless.start() can be called immediately.
# RunPod health check passes as soon as start() is called (~5s), regardless of model load time.
# Requests block on _model_ready until loading completes.

_model_ready = threading.Event()
_model_error: Exception | None = None
processor = None
model = None
tokenizer = None


def _load_model():
    global processor, model, tokenizer, _model_error
    try:
        t0 = time.time()
        print(f"[glm-ocr] Loading processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
        )
        print(f"[glm-ocr] Loading model weights...")
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
        # separate CUDA Conv3d kernel. This is the known perf bottleneck:
        # ~30s/image on H100 with Conv3d vs ~1s with linear.
        # See: https://huggingface.co/zai-org/GLM-OCR/discussions/36
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
                print("[glm-ocr] Applied Conv3d→linear monkey patch (22k patch speedup)")
            else:
                print("[glm-ocr] WARN: Could not find visual.patch_embed — skipping monkey patch")
        except Exception as mp_err:
            print(f"[glm-ocr] WARN: Monkey patch failed (non-fatal): {mp_err}")

        elapsed = time.time() - t0
        print(f"[glm-ocr] Model ready in {elapsed:.1f}s. Device map: {model.hf_device_map}")
    except Exception as e:
        _model_error = e
        print(f"[glm-ocr] ERROR loading model: {e}")
    finally:
        _model_ready.set()


_load_thread = threading.Thread(target=_load_model, daemon=True, name="model-loader")
_load_thread.start()


def decode_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    img_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _log_memory():
    """Log GPU and system memory usage for diagnostics."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[glm-ocr] GPU mem: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"[glm-ocr] System RAM: {vm.used/1024**3:.1f}GB/{vm.total/1024**3:.1f}GB ({vm.percent}%)")
    except ImportError:
        pass


def ocr_single_image(image: Image.Image, prompt: str = "OCR:") -> str:
    """Run OCR on a single image and return extracted text."""
    t0 = time.time()
    model_device = next(model.parameters()).device
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model_device)

    _log_memory()
    print(f"[glm-ocr] Starting generate (input_ids shape: {inputs.get('input_ids', torch.tensor([])).shape})...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    output_ids = generated_ids[:, input_len:]
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"[glm-ocr] Generated {len(output_ids[0])} tokens in {elapsed:.1f}s")
    _log_memory()
    return text.strip()


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
    # Wait for model to finish loading (blocks only on first cold-start request)
    if not _model_ready.is_set():
        print("[glm-ocr] Waiting for model to finish loading...")
        _model_ready.wait()

    if _model_error is not None:
        return {"error": f"Model failed to load: {_model_error}"}

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
# Called immediately — before model finishes loading.
# RunPod health check passes here; requests block in handler() until _model_ready is set.
import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
