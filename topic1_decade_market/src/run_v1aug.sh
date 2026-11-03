#!/bin/bash
# N-1 on the text-decade adapter: v1 recipe with bandwidth-randomized targets (needs lora/codes_A_train_aug.npy from 101_make_aug_codes.py), then standard generation, three-branch guidance (ge 10) for the six decades, and the era-free bandwidth prompts. usage: run_v1aug.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name v1aug --r 16 --steps 3000 --bs 8 --lr 1e-4 --seed 0 --aug > logs/lora_v1aug.log 2>&1
GPU=$1 V=v1aug G=gen_lora_v1aug P=lora_v1aug ./run_lora_gen.sh
bash run_tri_ad.sh $1 lora 3 10 50 v1aug
bash run_tri_x2.sh $1 3 10 v1aug
AD=v1aug NS=gen_lora_v1aug bash src/run_tri_pbw.sh $1 10
touch v1aug_chain.done
