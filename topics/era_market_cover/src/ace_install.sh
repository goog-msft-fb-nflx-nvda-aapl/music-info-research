#!/bin/bash
cd ~/era_market; source venv_ace/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 || exit 1
pip install -q "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" scipy soundfile loguru einops "accelerate>=1.12.0" diskcache numba "vector-quantize-pytorch>=1.27.15" torchcodec "torchao>=0.16.0,<0.17.0" toml peft lycoris-lora lightning modelscope typer-slim pytorch-wavelets pywavelets "setuptools<72" librosa safetensors huggingface_hub
touch ace_install.done
