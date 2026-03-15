#!/usr/bin/env bash
# Neptun.sh — Handler script for the Neptun DNBN research framework.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
TMP_DIR="$SCRIPT_DIR/tmp"

usage() {
    cat <<EOF
Neptun — Distributed Neural and Bonds Network (DNBN)

Usage:
  ./Neptun.sh --install                                Install venv and download datasets
  ./Neptun.sh --clean                                  Remove venv and tmp data
  ./Neptun.sh --list                                   List available experiments
  ./Neptun.sh --run <dataset> --sys-dnbn <config>      Run train/predict experiment

Options:
  --device <cpu|cuda|mps|auto>   Computation device (default: auto)

Examples:
  ./Neptun.sh --install
  ./Neptun.sh --run mnist --sys-dnbn configs/sys_dnbn_mnist_2node.json
  ./Neptun.sh --run cifar10 --sys-dnbn configs/sys_dnbn_cifar10_2node.json
  ./Neptun.sh --list
  ./Neptun.sh --clean
EOF
}

cmd_install() {
    echo "============================================"
    echo "  Neptun: Installing environment"
    echo "============================================"

    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
        echo "Run ./Neptun.sh --clean first to reinstall."
        exit 1
    fi

    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements.txt"

    echo ""
    echo "============================================"
    echo "  Downloading datasets"
    echo "============================================"
    mkdir -p "$TMP_DIR"
    cd "$SCRIPT_DIR"
    python -m neptun.datasets --download

    echo ""
    echo "============================================"
    echo "  Installation complete"
    echo "============================================"
}

cmd_clean() {
    echo "============================================"
    echo "  Neptun: Cleaning"
    echo "============================================"

    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        echo "Removed: $VENV_DIR"
    else
        echo "No venv found."
    fi

    if [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
        echo "Removed: $TMP_DIR"
    else
        echo "No tmp directory found."
    fi

    echo "Clean complete."
}

cmd_list() {
    _activate
    cd "$SCRIPT_DIR"
    python -m neptun --list
}

cmd_run() {
    _activate
    cd "$SCRIPT_DIR"
    python -m neptun "$@"
}

_activate() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: Virtual environment not found."
        echo "Run ./Neptun.sh --install first."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
}

# ── Argument parsing ──────────────────────────────────────────

if [ $# -eq 0 ]; then
    usage
    exit 0
fi

case "$1" in
    --install)
        cmd_install
        ;;
    --clean)
        cmd_clean
        ;;
    --list)
        cmd_list
        ;;
    --run)
        cmd_run "$@"
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        echo ""
        usage
        exit 1
        ;;
esac
