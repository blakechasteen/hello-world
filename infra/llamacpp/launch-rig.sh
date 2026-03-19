#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# llama-server launcher for mining rig (3x RTX 3080)
#
# For quick start/stop without systemd. For production, use the
# systemd service (llamacpp-rig.service) instead — see AP-11.
#
# Usage:
#   ./launch-rig.sh start     # Start server
#   ./launch-rig.sh stop      # Stop server
#   ./launch-rig.sh restart   # Restart
#   ./launch-rig.sh health    # Check health
#   ./launch-rig.sh status    # PIDs + health
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/.llamacpp_pid"

# --- Configuration ---
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen3.5-30b-q4_k_m.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-4}"

mkdir -p "$LOG_DIR"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "llama-server already running (PID $(cat "$PID_FILE"))"
        healthcheck
        return
    fi

    echo "Starting llama-server..."
    echo "  Model:    $MODEL"
    echo "  Port:     $PORT"
    echo "  Context:  $CTX_SIZE"
    echo "  Parallel: $PARALLEL"
    echo "  GPUs:     0,1,2 (tensor-split 12,12,12)"
    echo ""

    CUDA_VISIBLE_DEVICES=0,1,2 \
    nohup "$LLAMA_SERVER" \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --ctx-size "$CTX_SIZE" \
        --n-gpu-layers 99 \
        --split-mode layer \
        --tensor-split 12,12,12 \
        --flash-attn \
        --parallel "$PARALLEL" \
        --cont-batching \
        --metrics \
        > "${LOG_DIR}/llamacpp.log" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Started (PID $(cat "$PID_FILE"))"
    echo ""

    echo "Waiting for server..."
    for i in $(seq 1 30); do
        sleep 2
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "llama-server ready on port $PORT"
            return
        fi
        printf "."
    done
    echo ""
    echo "WARNING: health check didn't pass within 60s. Check ${LOG_DIR}/llamacpp.log"
}

stop() {
    echo "Stopping llama-server..."
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  Stopped PID $pid"
        else
            echo "  PID $pid not running"
        fi
        rm "$PID_FILE"
    else
        echo "  No PID file found"
    fi
}

healthcheck() {
    status=$(curl -s "http://localhost:${PORT}/health" 2>/dev/null || echo '{"status":"unreachable"}')
    echo "Health: $status"
}

status() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "llama-server running (PID $pid)"
        else
            echo "PID file exists but process $pid not running"
        fi
    else
        echo "No PID file"
    fi
    healthcheck
}

case "${1:-start}" in
    start)       start ;;
    stop)        stop ;;
    restart)     stop; sleep 2; start ;;
    health)      healthcheck ;;
    status)      status ;;
    *)           echo "Usage: $0 {start|stop|restart|health|status}" ;;
esac
