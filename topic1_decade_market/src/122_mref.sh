#!/bin/bash
# T-4m: reference-audio market transfer, strengths 0.5 and 0.8, then 37_eval_market.py read-out. usage: bash 122_mref.sh <gpu>   (from ~/era_market)
cd ~/era_market; g=$1
for k in "ace_mref 0.5" "ace_mref08 0.8"; do set -- $k
  (cd ext/ACE-Step-1.5; GENDIR=$1 CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/122_ace_reference_market.py --strength $2 > ~/era_market/logs/$1.log 2>&1)
  GENDIR=$1 CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/37_eval_market.py --out results/$1_eval.json --seeds 0 > logs/$1_eval.log 2>&1
done
touch mref.done
