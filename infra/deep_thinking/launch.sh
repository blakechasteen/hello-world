#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Deep Thinking GPU Server Launcher
# Starts 3 AirLLM instances, one per RTX 3080.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PIDS_FILE="${SCRIPT_DIR}/.gpu_server_pids"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-108B}"
COMPRESSION="${COMPRESSION:-4bit}"
HOST="${HOST:-0.0.0.0}"

mkdir -p "$LOG_DIR"

start() {
    echo "Starting 3 AirLLM GPU servers..."
    echo "  Model:       $MODEL_ID"
    echo "  Compression: $COMPRESSION"
    echo ""

    > "$PIDS_FILE"

    for gpu_id in 0 1 2; do
        port=$((8001 + gpu_id))
        log="${LOG_DIR}/gpu${gpu_id}.log"

        echo "  GPU ${gpu_id} → port ${port} (log: ${log})"

        CUDA_VISIBLE_DEVICES=$gpu_id \
        GPU_ID=$gpu_id \
        MODEL_ID=$MODEL_ID \
        COMPRESSION=$COMPRESSION \
        PORT=$port \
            nohup python -m uvicorn gpu_server:app \
                --host "$HOST" \
                --port "$port" \
                --app-dir "$SCRIPT_DIR" \
                > "$log" 2>&1 &

        echo $! >> "$PIDS_FILE"
    done

    echo ""
    echo "Waiting for servers to initialize..."
    sleep 10

    healthcheck
}

stop() {
    echo "Stopping GPU servers..."
    if [ -f "$PIDS_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "  Stopped PID $pid"
            fi
        done < "$PIDS_FILE"
        rm "$PIDS_FILE"
    else
        echo "  No PID file found."
    fi
}

healthcheck() {
    echo "Health check:"
    for gpu_id in 0 1 2; do
        port=$((8001 + gpu_id))
        status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
        if [ "$status" = "200" ]; then
            echo "  GPU ${gpu_id} (port ${port}): ✓ healthy"
        else
            echo "  GPU ${gpu_id} (port ${port}): ✗ unreachable (HTTP ${status})"
        fi
    done
}

status() {
    if [ -f "$PIDS_FILE" ]; then
        echo "GPU server PIDs:"
        cat "$PIDS_FILE"
        echo ""
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
