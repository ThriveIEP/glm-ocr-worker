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

# Model revision for trust_remote_code safety.
# trust_remote_code=True executes model Python files from HF Hub at load time.
# Pinning to a specific commit makes the executed code immutable.
# Default is "main" — override with --build-arg MODEL_REVISION=<sha> before production.
ARG MODEL_REVISION=main

# NOTE: Model weights are downloaded at container startup, not baked into the image.
# RunPod's build servers time out on large (~2GB) HuggingFace downloads.
# The HF cache (~/.cache/huggingface) persists within a warm worker's lifetime,
# so subsequent requests on the same worker pay no download cost.
# Write build provenance so the image is traceable.
RUN printf 'TRANSFORMERS_REF=%s\nMODEL_REVISION=%s\n' "${TRANSFORMERS_REF}" "${MODEL_REVISION}" > /app/build-provenance.txt

# Copy handler
COPY handler.py .

# Expose MODEL_REVISION to handler.py at runtime so it loads the correct revision.
ENV MODEL_REVISION=${MODEL_REVISION}

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
