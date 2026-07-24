#!/bin/bash
# Reviewer-rebuttal experiment: CodeT5+ 770M, 5 tasks (4 main-paper tasks + code_repair).
# Adapters injected into BOTH encoder and decoder (code_repair needs decoder adaptation).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

EPOCHS=5
MAXN=300000
LR=1e-4
BDIM=64
BATCH=8
GPU=1,2

echo "=== CodeT5+ 5-task (code_repair) — $(date) ==="
echo "    epochs=$EPOCHS  max_train_samples=$MAXN  batch=$BATCH  bottleneck_dim=$BDIM  GPU=$GPU"
echo ""

conda run --no-capture-output -n multitask \
    env CUDA_VISIBLE_DEVICES="${GPU}" python -u run_repair.py \
        --bottleneck_dim $BDIM \
        --max_train_samples $MAXN \
        --num_train_epochs $EPOCHS \
        --train_batch_size $BATCH --eval_batch_size $BATCH \
        --code_length 512 --nl_length 128 \
        --learning_rate $LR --max_grad_norm 1.0 \
        --sampling_temperature 0.3 \
        --repair_eval_cap_batches 20 \
        --output_model_name codet5p_repair_5task_adapter_norm_v2 \
        2>&1 | tee logs/codet5p_repair_5task_adapter_norm_v2.log

echo ""
echo "=== Done — $(date) ==="
