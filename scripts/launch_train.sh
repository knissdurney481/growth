#!/usr/bin/env bash
set -euo pipefail
SHEET="${1:-configs/experiment/main.yaml}"
torchrun --standalone --nproc_per_node=4 -m growth_twin.cartographer chart --sheet "${SHEET}"
