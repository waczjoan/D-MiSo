#!/bin/bash

CONDA_CUDA=true

for arg in "$@"; do
  if [ "$arg" == "--conda-cuda" ]; then
    CONDA_CUDA=true
  fi
done

conda create -n dmiso python=3.8 pip -y

if [ "$CONDA_CUDA" = true ]; then
  conda install -n dmiso cuda -c nvidia/label/cuda-11.8.0 -y
fi

conda run -n dmiso pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda run -n dmiso pip install submodules/diff-gaussian-rasterization
conda env update -n dmiso --file environment.yml