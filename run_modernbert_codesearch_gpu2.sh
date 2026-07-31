#!/bin/bash
# ModernBERT single-task (code_search) — sequential PEFT sweep on GPU 2
# Order: lora -> serial adapter -> parallel adapter -> full finetuning
set -euo pipefail

mkdir -p logs

EPOCHS=10
LR=1e-4
BATCH=32
LORA_R=32
BDIM=64
MAXN=300000
TASK="code_search"
GPU=2
MODEL="answerdotai/ModernBERT-base"

run_model() {
    local PEFT="$1"
    local LABEL="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
            --model_name_or_path "$MODEL" \
            --peft_module "$PEFT" \
            --tasks $TASK \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size $BATCH --eval_batch_size $BATCH \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --output_model_name "$LABEL" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "logs/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

echo "=== ModernBERT code_search PEFT sweep (GPU${GPU}) — $(date) ==="

run_model lora             modernbert_codesearch_lora             --lora_r $LORA_R
run_model adapter          modernbert_codesearch_serialAdapter    --bottleneck_dim $BDIM
run_model parallel_adapter modernbert_codesearch_parallelAdapter  --bottleneck_dim $BDIM
run_model full             modernbert_codesearch_full

echo ""
echo "=== ModernBERT code_search PEFT sweep complete (GPU${GPU}) — $(date) ==="
