#!/bin/bash
# N-5a extension: after run_bon16_gen.sh on GPU 0 and 1: probe evaluation and content metrics of the 16-seed pool, then the selection analysis (N = 1, 2, 4, 8, 16). usage: run_bon16_eval.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; n=gen_ax_ctrlc_ge10; t=ax_ctrlc_bon16
until [ -f bon16_gen_0.done ] && [ -f bon16_gen_1.done ]; do sleep 60; done
S16=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${t}_eval.json --train_cond acccodec_norm --seeds ${S16//,/ } --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${t}_eval.log 2>&1
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s SEEDS=$S16 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${t}_content.json > logs/${t}_content.log 2>&1
CUDA_VISIBLE_DEVICES= ./venv/bin/python src/103_best_of_n.py $t 16 > logs/${t}_bon.log 2>&1
touch ${t}.done
