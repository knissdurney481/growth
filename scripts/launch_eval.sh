#!/usr/bin/env bash
set -euo pipefail
SHEET="${1:-configs/experiment/main.yaml}"
CKPT="${2:-runs/main/best.ckpt}"
python -m growth_twin.cartographer assess --sheet "${SHEET}" --checkpoint "${CKPT}"
