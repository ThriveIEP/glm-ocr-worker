"""RunPod Serverless handler for GLM-OCR (zai-org/GLM-OCR).

Accepts base64-encoded images, returns OCR text.
Uses transformers pipeline directly (0.9B model fits easily in 24GB VRAM).

Fast cold starts require a pre-populated RunPod Network Volume mounted at
/runpod-volume with HF_HOME=/runpod-volume/hf-cache set in the endpoint env.
Without the volume, model downloads at startup (~3 min cold start).
"""

import base64
import io
import os
import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# --- Startup diagnostics (logged to RunPod worker logs) ---

MODEL_NAME = os.environ.get("MODEL_NAME", "zai-org/GLM-OCR")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
hf_cache_dir = os.path.join(HF_HOME, "hub")

print(f"[glm-ocr] Starting worker — {MODEL_NAME}@{MODEL_REVISION}")
print(f"[glm-ocr] HF_HOME: {HF_HOME}")

volume_mounted = os.path.isdir("/runpod-volume")
cache_populated = os.path.isdir(hf_cache_dir) and bool(os.listdir(hf_cache_dir))

if volume_mounted and cache_populated:
    print("[glm-ocr] Network volume mounted + model cache found — fast cold start.")
elif volume_mounted and not cache_populated:
    print("[glm-ocr] WARNING: Network volume mounted but cache is empty.")
    print("[glm-ocr] Model will download now (~3 min). Pre-populate the volume to fix this.")
    print("[glm-ocr] Run once on a pod: HF_HOME=/runpod-volume/hf-cache python -c \"")
    print("[glm-ocr]   from transformers import AutoModelForCausalLM, AutoProcessor")
    print(f"[glm-ocr]   AutoProcessor.from_pretrained('{MODEL_NAME}', trust_remote_code=True)")
    print(f"[glm-ocr]   AutoModelForCausalLM.from_pretrained('{MODEL_NAME}', trust_remote_code=True)\"")
else:
    print("[glm-ocr] WARNING: No network volume at /runpod-volume.")
    print("[glm-ocr] Model will download to container disk (~3 min, lost on worker shutdown).")
    print("[glm-ocr] Attach a network volume at /runpod-volume for persistent caching.")

# --- Model loading (runs once at container start, persists across requests) ---

print(f"[glm-ocr] Loading model...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = processor.tokenizer
print(f"[glm-ocr] Model loaded. Device map: {model.hf_device_map}")


def decode_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    img_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def ocr_single_image(image: Image.Image, prompt: str = "OCR:") -> str:
    """Run OCR on a single image and return extracted text."""
    model_device = next(model.parameters()).device
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    output_ids = generated_ids[:, input_len:]
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
import runpod  # noqa: E402

runpod.serverless.start({"handler": handler})
