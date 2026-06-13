#!/bin/bash
# Pairwise: flakiness_detect + code_search — serial adapter (d=64), norm_v2
# GPU 0 (sequential): UniXcoder → ModernBERT → CodeT5+ → DeepSeek → Qwen
set -euo pipefail

mkdir -p logs/pairwise

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64
TASKS="flakiness_detect,code_search"
LOSS="normalized"
GPU=0

run_model() {
    local MODEL_PATH="$1"
    local LABEL="$2"
    local BATCH="$3"

    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
            --model_name_or_path "$MODEL_PATH" \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size "$BATCH" --eval_batch_size "$BATCH" \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$LABEL" \
            2>&1 | tee "logs/pairwise/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

echo "=== Pairwise flak+search — $(date) ==="
echo "    tasks=$TASKS  peft=adapter(d=$BDIM)  loss=$LOSS  epochs=$EPOCHS"
echo "    GPU${GPU} (sequential): UniXcoder → ModernBERT → CodeT5+ → DeepSeek → Qwen"
echo ""

run_model microsoft/unixcoder-base            unixcoder_flak_search_adapter_norm_v2   32
run_model answerdotai/ModernBERT-base          modernbert_flak_search_adapter_norm_v2  32
run_model Salesforce/codet5p-770m              codet5p_flak_search_adapter_norm_v2     16
run_model deepseek-ai/deepseek-coder-1.3b-base deepseek_flak_search_adapter_norm_v2    8
run_model Qwen/Qwen2.5-Coder-1.5B             qwen25_flak_search_adapter_norm_v2       8

echo ""
echo "=== Pairwise flak+search complete — $(date) ==="
