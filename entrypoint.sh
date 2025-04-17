#!/usr/bin/env bash
set -e

# Download I2V-14B-480P weights if not already present
echo "Checking for I2V-14B-480P model weights..."
# Use huggingface_hub to download snapshot
MODEL_DIR=$(python - << 'PYCODE'
from huggingface_hub import snapshot_download
print(snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P"))
PYCODE
)
echo "Model weights are available at ${MODEL_DIR}"

# Start the server with specified parameters
echo "Starting server..."
exec python server.py \
    --ckpt_dir "${MODEL_DIR}" \
    --task i2v-14B \
    --frame_num 81 \
    --sample_steps 30 \
    --t5_cpu \
    --enable_teacache \
    --use_ret_steps