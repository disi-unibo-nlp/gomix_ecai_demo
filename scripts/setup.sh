#!/usr/bin/env bash
set -euo pipefail

# Check if Python 3.11 is available via py launcher
if ! py -3.11 --version &> /dev/null; then
    echo "ERROR: Python 3.11 is not installed."
    echo "Please install Python 3.11.9 from https://www.python.org/downloads/release/python-3119/"
    exit 1
fi

# Verify Python 3.11 version
py -3.11 --version

# Use Python 3.11 explicitly for all pip commands
py -3.11 -m pip install --upgrade pip setuptools wheel
py -3.11 -m pip install \
  torch==2.0.1 \
  numpy==1.26.4 \
  typing_extensions==4.12.2 \
  typing-inspection==0.4.2 \
  gradio \
  pandas \
  matplotlib

# Install PyTorch Geometric and its extensions with CUDA 11.8 support
py -3.11 -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
py -3.11 -m pip install -r "$script_dir/../gomix/requirements.txt"