#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel

pip install \
  torch==2.0.1 \
  numpy==1.26.4 \
  typing_extensions==4.12.2 \
  typing-inspection==0.4.2 \
  gradio \
  pandas \
  matplotlib

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
pip install -r "$script_dir/../gomix/requirements.txt"
