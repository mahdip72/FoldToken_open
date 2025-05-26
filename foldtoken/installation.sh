#!/usr/bin/env bash
set -euo pipefail

# Update build basics
python -m pip install -U pip setuptools wheel packaging

# ---------------------------------------------------------------------
# 1) PyTorch 2.0.1 + CUDA 11.7 (official wheels)
# ---------------------------------------------------------------------
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
        --index-url https://download.pytorch.org/whl/cu117

# ---------------------------------------------------------------------
# 2) Core numerics that some wheels depend on
# ---------------------------------------------------------------------
pip install "scipy>=1.11,<1.12"

# ---------------------------------------------------------------------
# 3) PyTorch-Geometric CUDA wheels (Torch 2.0.1 + CU117)
#    Hosted on data.pyg.org — disable PyPI so we don’t grab CPU builds.
# ---------------------------------------------------------------------
PYG_INDEX="https://data.pyg.org/whl/torch-2.0.1+cu117.html"

pip install --no-index --find-links "$PYG_INDEX" torch-scatter==2.1.2+pt20cu117
pip install --no-index --find-links "$PYG_INDEX" torch-sparse==0.6.17+pt20cu117
pip install --no-index --find-links "$PYG_INDEX" torch-cluster==1.6.3+pt20cu117
pip install --no-index --find-links "$PYG_INDEX" torch-spline-conv==1.2.2+pt20cu117
pip install --no-index --find-links "$PYG_INDEX" pyg-lib==0.2.0+pt20cu117
pip install --find-links https://data.pyg.org/whl/torch-2.0.1+cu117.html torch-geometric==2.3.1

# ---------------------------------------------------------------------
# 4) Optional extras and utilities
# ---------------------------------------------------------------------
pip install flash-attn --no-build-isolation         # Ampere+ GPUs
pip install pytorch-lightning==1.9.0                       # training framework
pip install omegaconf==2.3.0                               # config library
pip install tqdm
