#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
until [ -f tri_lora_m3e7k250.done ] && [ -f tri2_vinet.done ]; do sleep 60; done
./venv/bin/python src/54_vinet_identity.py gen_tri_lora_m3e7k250 > logs/tri3_vinet.log 2>&1
touch tri3_vinet.done
