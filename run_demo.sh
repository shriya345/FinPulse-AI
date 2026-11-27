#!/bin/bash
# Local demo launcher for the verified Apple Silicon environment.
set -euo pipefail
cd "$(dirname "$0")"
runtime="$PWD/.runtime"
py="${FINPULSE_PYTHON:-/opt/anaconda3/bin/python}"
mkdir -p "$runtime"
export DATABASE_URL="${DATABASE_URL:-sqlite:///$runtime/demo.db}"
export TELEMETRY_PATH="${TELEMETRY_PATH:-$runtime/dashboard.jsonl}"
model_pid=""
cleanup() {
  if [[ -n "$model_pid" ]]; then kill "$model_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
"$py" pipeline.py demo
if [[ "${1:-}" != "--offline" ]]; then
  if ! "$py" -c 'from live import installed_models; raise SystemExit(not installed_models())'; then
    if [[ ! -x "$runtime/build/bin/llama-server" || ! -f "$runtime/qwen.gguf" ]]; then
      echo "Run bash setup_local_model.sh to prepare the local model, or bash run_demo.sh --offline."
      exit 1
    fi
    "$runtime/build/bin/llama-server" -m "$runtime/qwen.gguf" --alias Qwen2.5-1.5B-Instruct-Q4_K_M --host 127.0.0.1 --port 18536 --threads 4 --ctx-size 4096 --parallel 1 > "$runtime/model.log" 2>&1 &
    model_pid=$!
    for ((tries=0; tries<60; tries++)); do
      if curl --noproxy 127.0.0.1 -fsS http://127.0.0.1:18536/health >/dev/null 2>&1; then break; fi
      kill -0 "$model_pid" 2>/dev/null || { echo "Model startup failed. See .runtime/model.log"; exit 1; }
      sleep 1
    done
    "$py" -c 'from live import installed_models; assert installed_models(), "Local model startup timed out"'
  fi
else
  export FINPULSE_OFFLINE=1
fi
"$py" -m streamlit run dashboard.py --server.address 127.0.0.1 --server.fileWatcherType none
