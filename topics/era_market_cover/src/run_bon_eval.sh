#!/bin/bash
# N-5a: after run_bon_gen.sh on all eras: probe evaluation and content metrics of the 8-seed pool, then the selection analysis. usage: run_bon_eval.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; n=gen_ax_ctrlc_ge10; t=ax_ctrlc_bon
until [ -f bon_gen_0.done ] && [ -f bon_gen_1.done ] && [ -f bon_gen_2.done ]; do sleep 60; done
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${t}_eval.json --train_cond acccodec_norm --seeds 0 1 2 3 4 5 6 7 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${t}_eval.log 2>&1
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s SEEDS=0,1,2,3,4,5,6,7 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${t}_content.json > logs/${t}_content.log 2>&1
CUDA_VISIBLE_DEVICES= ./venv/bin/python src/103_best_of_n.py $t 8 > logs/${t}_bon.log 2>&1
touch ${t}.done
