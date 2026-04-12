"""RunPod Serverless handler for GLM-OCR (zai-org/GLM-OCR).

Accepts base64-encoded images, returns OCR text.
Uses transformers pipeline directly (0.9B model fits easily in 24GB VRAM).
"""

import base64
import io
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# --- Model loading (runs once at container start, persists across requests) ---

MODEL_NAME = os.environ.get("MODEL_NAME", "zai-org/GLM-OCR")

# FIX-4: Pin to a known-good HF commit for trust_remote_code safety.
# With trust_remote_code=True, from_pretrained() executes model Python files from HF Hub.
# Without a pin, a compromised or updated zai-org/GLM-OCR repo could inject arbitrary code.
# To pin: find the commit SHA at https://huggingface.co/zai-org/GLM-OCR/commits/main,
# then set MODEL_REVISION env var in the RunPod endpoint config (or Dockerfile ARG).
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")

print(f"Loading {MODEL_NAME}@{MODEL_REVISION} ...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # FIX-2: was device_map=DEVICE (bare string is invalid accelerate API)
    trust_remote_code=True,
)
# FIX-5: Use the tokenizer embedded in the processor rather than a separate AutoTokenizer load.
# A separate load may resolve to a different vocab or special token mapping for VLMs.
tokenizer = processor.tokenizer
print(f"Model loaded. Device map: {model.hf_device_map}")


def decode_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    if image_data.startswith("data:"):
        # Strip data URI prefix (e.g., "data:image/png;base64,...")
        image_data = image_data.split(",", 1)[1]
    img_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def ocr_single_image(image: Image.Image, prompt: str = "OCR:") -> str:
    """Run OCR on a single image and return extracted text."""
    # FIX-2: Move inputs to the actual model device (handles device_map="auto" placement)
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

    # Decode only the generated tokens (skip the input tokens).
    # Defensive fallback if processor omits input_ids (non-standard VLM processors).
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

    Note: field name is "response" (not "text") to match glm-ocr-service.ts expectation.
    """
    # FIX-6: Wrap entire handler in try/except so unhandled exceptions (CUDA OOM,
    # corrupt input, tokenizer errors) return a structured error instead of crashing
    # the worker process.
    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "OCR:")

        # Single image
        if "image" in job_input:
            image = decode_image(job_input["image"])
            text = ocr_single_image(image, prompt)
            return {"response": text}  # FIX-1: was "text", must be "response" (matches server)

        # Multiple images (batch)
        if "images" in job_input:
            results = []
            for i, img_data in enumerate(job_input["images"]):
                # FIX-6: Per-image error isolation — one bad page doesn't abort the batch
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
