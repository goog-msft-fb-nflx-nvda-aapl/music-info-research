#!/bin/bash
# Discogs-EffNet read-out (CPU, venv_ess) of the kept generation directories of 2026-11-08
cd ~/era_market; export OMP_NUM_THREADS=8
E6=neutral,1960s,1970s,1980s,1990s,2000s,2010s
./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_ax_onehot_pfx16_ge10_hpcp --seeds 0 1 2 --eras $E6 --out results/discogs_decade_ax_onehot_pfx16_hpcp.json > logs/effnet_pfx16_hpcp.log 2>&1
./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_ax_ctrlc_ge10_hpcp --seeds 0 1 2 --eras $E6 --out results/discogs_decade_ax_ctrlc_hpcp.json > logs/effnet_ctrlc_hpcp.log 2>&1
./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_ax_onehot_pfx16_ge10_hpcp_dose --seeds 0 1 2 --eras a0,a0.2,a0.4,a0.6,a0.8,a1 --out results/discogs_decade_dose.json > logs/effnet_dose.log 2>&1
./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_ax_onehot_pfx16_ge10_hpcp_dose2 --seeds 0 1 2 --eras a0.3,a0.35,a0.45,a0.5,a0.55 --out results/discogs_decade_dose2.json > logs/effnet_dose2.log 2>&1
touch effnet.done
