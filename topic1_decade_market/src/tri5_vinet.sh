#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4
until [ -f tri_lora_m3e7v1s1.done ] && [ -f tri_lora_m3e10v1s1.done ] && [ -f tri_lora_m3e7v1s2.done ] && [ -f tri_lora_m3e10v1s2.done ]; do sleep 60; done
./venv/bin/python src/54_vinet_identity.py gen_tri_lora_m3e7v1s1 gen_tri_lora_m3e10v1s1 gen_tri_lora_m3e7v1s2 gen_tri_lora_m3e10v1s2 > logs/tri5_vinet.log 2>&1
touch tri5_vinet.done
