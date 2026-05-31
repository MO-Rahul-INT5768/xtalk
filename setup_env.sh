#!/bin/bash
# One-time setup: installs all dependencies into the persistent 'voxindic' conda env.
# Run this ONCE after creating the env, or if packages are missing.
# The voxindic env lives at /home/sagemaker-user/.conda/envs/voxindic (persists across restarts).

set -e
PYTHON=/home/sagemaker-user/.conda/envs/voxindic/bin/python
PIP="$PYTHON -m pip"

echo "=== Installing xtalk + base deps ==="
cd /home/sagemaker-user/xtalk
$PIP install -e ".[paraformer-local,edge-tts,server]" --quiet

echo "=== Pinning starlette to compatible version ==="
$PIP install "starlette==0.35.1" --quiet

echo "=== Installing faster-whisper + soxr ==="
$PIP install faster-whisper soxr --quiet

echo "=== Installing vLLM (takes a few minutes) ==="
$PIP install vllm --quiet

echo "=== Installing numpy<2 for vllm compatibility ==="
$PIP install "numpy<2" --quiet

echo ""
echo "All done! Run ./start.sh to start the services."
