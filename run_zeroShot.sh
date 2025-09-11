#!/usr/bin/env bash
set -e

python3 zero_shot.py \
    --modelName codellama/CodeLlama-34b-Instruct-hf \
    --seed 42 \
    --run_clone \
    --clone_code_file ./datasets/dataset_clone/data.jsonl \
    --clone_pairs_file ./datasets/dataset_clone/test.txt \
    --cs_topk 10 2>&1 | tee zeroShot_results_cloneMistral.log


    #  --run_clone \
    #--clone_code_file ./datasets/dataset_clone/code.jsonl \
    #--clone_pairs_file ./datasets/dataset_clone/test.txt \
    #--run_codesearch \
    #--codesearch_file ./datasets/code_search/test.jsonl \
    #--run_vuln \
    #--vuln_file ./datasets/dataset_vulnerabilty/test.jsonl \
    #--run_flaky \
    #--flaky_file ./datasets/dataset_flakytest/valid.json \