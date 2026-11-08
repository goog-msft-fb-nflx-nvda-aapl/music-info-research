#!/bin/bash
# era x language grid with 3 generation seeds (T-10b): lyric-route vocals for dataset A at seeds 1 and 2 (ACE-Step cover mode strength 0.95, same translations as seed 0), stem swap, then 151_two_axis_grid.py over seeds 0,1,2 (accompaniment seed s with vocal seed s). usage: bash src/run_grid3.sh <gpu_seed1> <gpu_seed2> (from ~/era_market)
cd ~/era_market; export TEXT=lyrics_scale_text_A.json DS=A
for sd in 1 2; do g=$([ $sd = 1 ] && echo $1 || echo $2)
 ( (cd ext/ACE-Step-1.5; CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/111_ace_lyric_route.py --strength 0.95 --seeds $sd --outdir ace_a_sd$sd > ~/era_market/logs/ace_a_sd$sd.log 2>&1)
   SEED=$sd SRC=ace_a_sd$sd DST=ace_a_swap_sd$sd CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/119_stem_swap_route.py > logs/two_axis_swap_sd$sd.log 2>&1 ) &
done; wait
SEEDS=0,1,2 OUTN=two_axis_grid_3seeds CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/151_two_axis_grid.py > logs/two_axis_grid3.log 2>&1
touch grid3.done
