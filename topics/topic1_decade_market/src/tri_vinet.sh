#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4
until [ -f tri_base_m3e3.done ] && [ -f tri_base_m3e5.done ] && [ -f tri_lora_m3e5.done ] && [ -f tri_lora_m3e7.done ]; do sleep 60; done
./venv/bin/python src/54_vinet_identity.py gen_tri_base_m3e3 gen_tri_base_m3e5 gen_tri_lora_m3e5 gen_tri_lora_m3e7 > logs/tri_vinet.log 2>&1
touch tri_vinet.done
