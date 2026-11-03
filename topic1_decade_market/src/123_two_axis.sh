#!/bin/bash
# T-10: two-axis composition (era-axis accompaniment x lyric-route language) on the 24 evaluation songs of dataset A. usage: bash 123_two_axis.sh <gpu> (from ~/era_market)
cd ~/era_market; g=$1; export TEXT=lyrics_scale_text_A.json DS=A
CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/121_lyric_scale_text.py > logs/two_axis_text.log 2>&1
(cd ext/ACE-Step-1.5; CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/111_ace_lyric_route.py --strength 0.95 --outdir ace_a > ~/era_market/logs/ace_a.log 2>&1)
SRC=ace_a DST=ace_a_swap CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/119_stem_swap_route.py > logs/two_axis_swap.log 2>&1
CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/123_two_axis.py > logs/two_axis.log 2>&1
touch two_axis.done
