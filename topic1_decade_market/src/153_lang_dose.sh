#!/bin/bash
# Language dose (market axis, continuous through the share of target-language lyrics): code-switched lyrics es/de at 25/50/75 % -> ACE-Step cover (strength 0.95, seeds 0-2) -> stem swap -> read-outs. usage: bash src/153_lang_dose.sh <gpu> (from ~/era_market)
cd ~/era_market; g=$1; export DS=A TEXT=lyrics_lang_dose_A.json
./venv/bin/python src/153_lang_dose_text.py > logs/lang_dose_text.log 2>&1
CONDS=$(for L in es de; do for f in 25 50 75; do printf "$L$f,"; done; done | sed 's/,$//')
(cd ext/ACE-Step-1.5; CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/111_ace_lyric_route.py --strength 0.95 --seeds 0 1 2 --outdir ace_ld > ~/era_market/logs/ace_ld.log 2>&1)
for sd in 0 1 2; do SEED=$sd CONDS=$CONDS SRC=ace_ld DST=ace_ld_swap CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/119_stem_swap_route.py > logs/lang_dose_swap$sd.log 2>&1; done
CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/153_lang_dose_eval.py > logs/lang_dose_eval.log 2>&1
touch lang_dose.done
