"""RunPod Serverless handler for GLM-OCR (zai-org/GLM-OCR).

Accepts base64-encoded images, returns OCR text.
Uses transformers pipeline directly (0.9B model fits easily in 24GB VRAM).

Cold start behavior:
- Worker signals ready immediately via runpod.serverless.start()
- Heavy imports (torch, transformers) and model download happen in background
- First request blocks until model is ready; subsequent requests are fast
"""

import sys
import os
import time

print(f"[glm-ocr] Python {sys.version}", flush=True)
print(f"[glm-ocr] Starting RunPod serverless worker...", flush=True)

# --- RunPod entrypoint FIRST ---
# Must call runpod.serverless.start() as early as possible.
# RunPod has a health check timeout — if start() isn't called quickly enough,
# the worker is killed. All heavy imports happen lazily in the handler.
import runpod

# State for lazy initialization
_initialized = False
_init_error = None
_model = None
_processor = None
_tokenizer = None


def _initialize():
    """Lazy initialization: import heavy libs + load model on first request."""
    global _initialized, _init_error, _model, _processor, _tokenizer

    if _initialized:
        return

    try:
        t0 = time.time()
        print("[glm-ocr] Importing torch...", flush=True)
        import torch
        import torch.nn.functional as F
        print(f"[glm-ocr] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[glm-ocr] GPU: {torch.cuda.get_device_name(0)}", flush=True)

        print("[glm-ocr] Importing transformers...", flush=True)
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"[glm-ocr] transformers {transformers.__version__}", flush=True)

        MODEL_NAME = os.environ.get("MODEL_NAME", "zai-org/GLM-OCR")
        MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
        HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        print(f"[glm-ocr] Loading {MODEL_NAME}@{MODEL_REVISION} (HF_HOME={HF_HOME})", flush=True)

        print("[glm-ocr] Loading processor...", flush=True)
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
        )

        print("[glm-ocr] Loading model weights...", flush=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _tokenizer = _processor.tokenizer

        # --- Monkey-patch Conv3d → F.linear for patch embedding ---
        try:
            base_model = _model.model if hasattr(_model, "model") else _model
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
        print(f"[glm-ocr] Model ready in {elapsed:.1f}s. Device map: {_model.hf_device_map}", flush=True)
        _initialized = True

    except Exception as e:
        _init_error = e
        _initialized = True  # Don't retry — report the error
        import traceback
        print(f"[glm-ocr] ERROR during initialization: {e}", flush=True)
        traceback.print_exc()


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
    import torch
    import base64
    import io
    from PIL import Image

    # Lazy init on first request
    _initialize()

    if _init_error is not None:
        return {"error": f"Model failed to load: {_init_error}"}

    try:
        job_input = event.get("input", {})
        prompt = job_input.get("prompt", "OCR:")

        def decode_image(image_data):
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]
            img_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        def ocr_single(image, prompt_text):
            t0 = time.time()
            model_device = next(_model.parameters()).device
            inputs = _processor(
                images=image, text=prompt_text, return_tensors="pt"
            ).to(model_device)
            with torch.no_grad():
                generated_ids = _model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False
                )
            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            output_ids = generated_ids[:, input_len:]
            text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
            elapsed = time.time() - t0
            print(f"[glm-ocr] Generated {len(output_ids[0])} tokens in {elapsed:.1f}s", flush=True)
            return text.strip()

        if "image" in job_input:
            image = decode_image(job_input["image"])
            text = ocr_single(image, prompt)
            return {"response": text}

        if "images" in job_input:
            results = []
            for i, img_data in enumerate(job_input["images"]):
                try:
                    image = decode_image(img_data)
                    text = ocr_single(image, prompt)
                    results.append({"index": i, "response": text})
                except Exception as img_err:
                    results.append({"index": i, "response": "", "error": str(img_err)})
            return {"results": results}

        return {"error": "No 'image' or 'images' field in input"}

    except Exception as e:
        torch.cuda.empty_cache()
        return {"error": str(e)}


print("[glm-ocr] Calling runpod.serverless.start()...", flush=True)
runpod.serverless.start({"handler": handler})
