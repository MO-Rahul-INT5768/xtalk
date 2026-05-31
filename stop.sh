#!/bin/bash
# Stop vLLM + xtalk server processes.

stop_proc() {
  local name=$1
  local pattern=$2
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null | tr '\n' ' ')
  if [ -z "$pids" ]; then
    echo "[$name] Not running."
  else
    echo "[$name] Killing PIDs: $pids"
    kill $pids 2>/dev/null
    sleep 2
    # Force-kill if still alive
    local remaining
    remaining=$(pgrep -f "$pattern" 2>/dev/null | tr '\n' ' ')
    if [ -n "$remaining" ]; then
      echo "[$name] Force-killing: $remaining"
      kill -9 $remaining 2>/dev/null
    else
      echo "[$name] Stopped."
    fi
  fi
}

stop_proc "xtalk"  "configurable_server.py"
stop_proc "vLLM"   "vllm.entrypoints.openai.api_server"
stop_proc "vLLM"   "VLLM::EngineCore"

echo ""
echo "=== All services stopped ==="
