#!/bin/bash
# server: re-embed every MERT condition after the inv_freq fix (mert_util.py), dataset $1 on GPU $2, then probes (04) for all stem conditions
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ds=$1
if [ $ds = A ]; then CONDS="mix_raw mix_norm voc_raw voc_norm acc_raw acc_norm recomb_norm acccodec_norm accharm_norm accperc_norm"; else CONDS="mix_raw mix_norm voc_raw voc_norm acc_raw acc_norm recomb_norm accharm_norm accperc_norm"; fi
./venv/bin/python src/02_embed.py --ds $ds --conds $CONDS > logs/reembed_$ds.log 2>&1
touch reembed_$ds.done
