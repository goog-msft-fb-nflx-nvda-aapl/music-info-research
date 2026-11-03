#!/bin/bash
# N-5b analysis of the later-stage SMC sets (CPU, stored embeddings): six-decade table, ordinal control, nuisance-controlled steering. usage: run_smc_post.sh   (waits for smc_ctrlc_N8_b{4,2}_f500.done written by run_smc.sh)
cd ~/era_market
while [ ! -f smc_ctrlc_N8_b4_f500.done ] || [ ! -f smc_ctrlc_N8_b2_f500.done ]; do sleep 60; done
L="smc_ctrlc_N8_b4_f500=results/smc_ctrlc_N8_b4_f500_eval.json smc_ctrlc_N8_b2_f500=results/smc_ctrlc_N8_b2_f500_eval.json smc_ctrlc_N8_b4=results/smc_ctrlc_N8_b4_eval.json smc_ctrlc_N8_b2=results/smc_ctrlc_N8_b2_eval.json"
OUT_PREFIX=results/six_decades_smc_f500 ./venv/bin/python src/78_six_decades.py $L > logs/six_smc_f500.log 2>&1
OUT_PREFIX=results/ordinal_control_smc_f500 ./venv/bin/python src/102_ordinal_control.py $L > logs/ord_smc_f500.log 2>&1
OUT_PREFIX=nuisance_steering_smc_f500 ./venv/bin/python src/94_nuisance_steering.py smc_ctrlc_N8_b4_f500 smc_ctrlc_N8_b2_f500 > logs/nuis_smc_f500.log 2>&1
touch smc_post_f500.done
