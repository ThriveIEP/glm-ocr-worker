# GLM-OCR RunPod Serverless Worker
# Serves zai-org/GLM-OCR (0.9B VLM) for document OCR
#
# Build: docker build -t glm-ocr-worker .
# Test:  docker run --gpus all -p 8080:8080 glm-ocr-worker
#
# To pin transformers to a specific commit (recommended once a stable SHA is identified):
#   docker build --build-arg TRANSFORMERS_REF=<commit-sha> -t glm-ocr-worker .
#   Browse SHAs at: https://github.com/huggingface/transformers/commits/main
#
# To pin the HuggingFace model revision (recommended for trust_remote_code safety):
#   docker build --build-arg MODEL_REVISION=<hf-commit-sha> -t glm-ocr-worker .
#   Browse SHAs at: https://huggingface.co/zai-org/GLM-OCR/commits/main

FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

# FIX-3: Parameterize transformers git ref for reproducible builds.
# GLM-OCR requires transformers>=5.3.0; PyPI stable is <5, so we install from git.
# Default is "main" — override at build time with --build-arg TRANSFORMERS_REF=<sha>
# to lock to a known-good commit and prevent silent regressions on rebuilds.
ARG TRANSFORMERS_REF=main

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    torch \
    torchvision \
    pillow \
    accelerate \
    sentencepiece \
    protobuf \
    && pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_REF}"

# FIX-4: Parameterize HF model revision for trust_remote_code safety.
# trust_remote_code=True executes model Python files from HF Hub at load time.
# Pinning to a specific commit makes the executed code immutable.
# Default is "main" — override with --build-arg MODEL_REVISION=<sha> before production.
ARG MODEL_REVISION=main

# Download and bake model weights into the image (~2GB)
# This eliminates cold-start model downloads (consensus best practice).
# Note: AutoTokenizer bake removed — handler uses processor.tokenizer (FIX-5).
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoProcessor; \
AutoProcessor.from_pretrained('zai-org/GLM-OCR', revision='${MODEL_REVISION}', trust_remote_code=True); \
AutoModelForCausalLM.from_pretrained('zai-org/GLM-OCR', revision='${MODEL_REVISION}', trust_remote_code=True); \
print('Model weights baked successfully')" \
    && printf 'TRANSFORMERS_REF=%s\nMODEL_REVISION=%s\n' "${TRANSFORMERS_REF}" "${MODEL_REVISION}" > /app/build-provenance.txt

# Copy handler
COPY handler.py .

# Expose MODEL_REVISION to handler.py at runtime so it loads the same revision
# that was baked into the image.
ENV MODEL_REVISION=${MODEL_REVISION}

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
