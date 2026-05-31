#!/bin/bash
# Start vLLM + xtalk server using the persistent voxindic conda env.
# Run this every time after a SageMaker space restart.

PYTHON=/home/sagemaker-user/.conda/envs/voxindic/bin/python
XTALK_DIR=/home/sagemaker-user/xtalk
LOG_DIR=$XTALK_DIR/logs
mkdir -p "$LOG_DIR"

# Make CUDA libraries (from base conda) visible to the voxindic env
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

# ── 1. Check vLLM is not already running ────────────────────────────────────
if /usr/bin/curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
  echo "[vLLM] Already running, skipping."
else
  echo "[vLLM] Starting Qwen2.5-7B-Instruct-AWQ..."
  VLLM_USE_FLASHINFER_SAMPLER=0 setsid $PYTHON -m vllm.entrypoints.openai.api_server \
    --model /home/sagemaker-user/model-store/stimm/llm/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.65 \
    --max-model-len 4096 \
    --attention-backend TRITON_ATTN \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 8000 \
    > "$LOG_DIR/vllm.log" 2>&1 &
  echo "[vLLM] Waiting for it to be ready (can take ~60s)..."
  for i in $(seq 1 60); do
    sleep 3
    if /usr/bin/curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
      echo "[vLLM] Ready!"
      break
    fi
    echo "  ... still loading ($((i*3))s)"
  done
fi

# ── 2. Check xtalk is not already running ───────────────────────────────────
if /usr/bin/curl -sf http://127.0.0.1:11995/ > /dev/null 2>&1; then
  echo "[xtalk] Already running, skipping."
else
  echo "[xtalk] Starting xtalk server on port 11995..."
  setsid $PYTHON $XTALK_DIR/examples/sample_app/configurable_server.py \
    --config $XTALK_DIR/examples/sample_app/local_sagemaker_opensource.json \
    --port 11995 \
    > "$LOG_DIR/xtalk_stdout.log" 2>&1 &
  sleep 8
  if /usr/bin/curl -sf http://127.0.0.1:11995/ > /dev/null 2>&1; then
    echo "[xtalk] Ready!"
  else
    echo "[xtalk] May still be loading. Check logs/xtalk_stdout.log"
  fi
fi

echo ""
echo "=== Services started ==="
echo "UI URL (replace domain):  https://<your-domain>/codeeditor/default/ports/11995/"
echo "Logs: $LOG_DIR/vllm.log  and  $LOG_DIR/xtalk_stdout.log"
