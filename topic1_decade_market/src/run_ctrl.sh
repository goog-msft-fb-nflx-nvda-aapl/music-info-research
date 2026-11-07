#!/bin/bash
# control for N-8: era 1990s only, 3 seeds, all 24 clips. usage: BETA=.. LAYERS=.. KROWS=.. [CALIB=1] bash run_ctrl.sh <gpu> <vector name> <tag>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; n=gen_ctrl_$3
./venv/bin/python src/146_steer_gen.py --name $2 --gm 3 --ge 10 --outdir $n --seeds 0 1 2 --eras 1990s > logs/ctrl_$3.log 2>&1
ERAS=1990s SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/ctrl_$3_content.json > logs/ctrl_$3_content.log 2>&1
ERAS=1990s GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/ctrl_$3_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/ctrl_$3_eval.log 2>&1
touch ctrl_$3.done
