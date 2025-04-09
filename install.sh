#!/bin/bash

conda create -n dmiso python=3.8 pip
conda activate dmiso
conda run -n dmiso pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda run -n dmiso pip install submodules/diff-gaussian-rasterization
conda env update -n dmiso --file environment.yml