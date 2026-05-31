#!/bin/bash
set -e
LOG_DIR=/home/sagemaker-user/xtalk/logs
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting vLLM..."
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
  --model /home/sagemaker-user/model-store/stimm/llm/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name Qwen2.5-7B-Instruct-AWQ \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.65 --max-model-len 2048 \
  --quantization awq --dtype float16 \
  --attention-backend TRITON_ATTN --enforce-eager \
  > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID"

echo "[$(date)] Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[$(date)] vLLM ready after ${i}s"
    break
  fi
  sleep 1
done

echo "[$(date)] Starting xtalk server..."
cd /home/sagemaker-user/xtalk
python examples/sample_app/configurable_server.py \
  --config examples/sample_app/local_sagemaker_opensource.json \
  --port 11995 \
  > "$LOG_DIR/xtalk.log" 2>&1 &
XTALK_PID=$!
echo "[$(date)] xtalk PID: $XTALK_PID"
echo "$VLLM_PID $XTALK_PID" > "$LOG_DIR/pids.txt"

wait $VLLM_PID $XTALK_PID
