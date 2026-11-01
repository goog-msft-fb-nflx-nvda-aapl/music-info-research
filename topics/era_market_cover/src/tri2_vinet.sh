#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4
until [ -f tri_lora_m3e10.done ] && [ -f tri_lora_m3e7.done ] && [ -f tri_lora_m5e7.done ] && [ -f qk_tri.done ]; do sleep 60; done
./venv/bin/python src/54_vinet_identity.py gen_tri_lora_m3e10 gen_tri_lora_m3e7 gen_tri_lora_m5e7 > logs/tri2_vinet.log 2>&1
touch tri2_vinet.done
