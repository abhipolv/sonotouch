#!/bin/bash

RECORDINGS_DIR="./recordings"
SONOSERIAL="sonoserial.py"
SONALYZE="sonalyze.py"

mkdir -p "$RECORDINGS_DIR"

function record_and_analyze() {
    echo "[INFO] Starting recording..."
    python3 "$SONOSERIAL" || exit 1

    latest_file=$(ls -t recording_*.wav 2>/dev/null | head -n 1)
    if [[ -z "$latest_file" ]]; then
        echo "[ERROR] No recording found."
        exit 1
    fi

    mv "$latest_file" "$RECORDINGS_DIR/"
    filepath="$RECORDINGS_DIR/$latest_file"

    echo "[INFO] Analyzing $filepath ..."
    python3 "$SONALYZE" "$filepath"
}

function clean_recordings() {
    echo "[INFO] Cleaning recordings..."
    rm -v "$RECORDINGS_DIR"/recording_*.wav 2>/dev/null || echo "[INFO] No recordings to delete."
}

case "$1" in
    run)
        record_and_analyze
        ;;
    clean)
        clean_recordings
        ;;
    *)
        echo "Usage: $0 {run|clean}"
        exit 1
        ;;
esac
