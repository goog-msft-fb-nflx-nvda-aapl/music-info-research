#!/bin/bash
# B-1: control-family adapter (SongEcho_base / SongEcho_large / MuseControlLite) generation, probe evaluation and content metrics. usage: run_sao.sh <gpu> <method> <tag>
cd ~/era_market; export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; g=$1; m=$2; tag=$3; ES=neutral,1960s,1970s,1980s,1990s,2000s,2010s
( cd ext/Text-to-Music_control_family; CUDA_VISIBLE_DEVICES=$g ../../venv_mcl/bin/python ../../src/100_songecho_sao.py --method $m --outdir gen_$tag > ../../logs/$tag.log 2>&1 )
CUDA_VISIBLE_DEVICES=$g ERAS=$ES GENDIR=gen_$tag ./venv/bin/python src/10_eval_gen.py --out results/${tag}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${tag}_eval.log 2>&1
ERAS=$ES SEEDS=0,1,2 GENDIR=gen_$tag ./venv/bin/python src/13_content_metrics.py --out results/${tag}_content.json > logs/${tag}_content.log 2>&1
touch $tag.done
