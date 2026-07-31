#!/bin/bash
set -euo pipefail
mkdir -p logs
conda run --no-capture-output -n multitask env CUDA_VISIBLE_DEVICES=0 python -u run_mlora.py \
    --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
    --lora_r 32 --mlora_B_num 4 --mlora_B_scale 1.0 \
    --loss_weighting normalized --tasks vul_detection,clone_detection,code_search,flakiness_detect \
    --max_train_samples 300000 --num_train_epochs 10 \
    --train_batch_size 8 --eval_batch_size 8 \
    --code_length 512 --nl_length 128 \
    --learning_rate 1e-4 --max_grad_norm 1.0 \
    --sampling_temperature 0.3 \
    --output_model_name deepseek_mlora_norm_v2_300k \
    2>&1 | tee logs/deepseek_mlora_norm_v2_300k.log
