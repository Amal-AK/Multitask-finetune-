#!/bin/bash
# Rerun failed clone+search models (unixcoder already done)
# OOM fixes: codet5p 16→8, deepseek 8→4
# ModernBERT fix: requires transformers>=4.48
#
#   GPU 0: ModernBERT
#   GPU 1: CodeT5+  (batch 8, was 16)
#   GPU 2: DeepSeek (batch 4, was 8)
#   GPU 3: Qwen     (batch 8, checkpoint from epoch 3 exists but restarting clean)
set -euo pipefail

mkdir -p logs/pairwise

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64
TASKS="clone_detection,code_search"
LOSS="normalized"

run_model() {
    local GPU="$1" MODEL="$2" LABEL="$3" BATCH="$4"
    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
            --model_name_or_path  "$MODEL" \
            --peft_module adapter  --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks "$TASKS" \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size "$BATCH" --eval_batch_size "$BATCH" \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$LABEL" \
            2>&1 | tee "logs/pairwise/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

echo "=== Pairwise clone+search rerun — $(date) ==="
echo "    GPU0: ModernBERT(32)  GPU1: CodeT5+(8)  GPU2: DeepSeek(4)  GPU3: Qwen(8)"
echo ""

run_model 0 answerdotai/ModernBERT-base            modernbert_clone_search_adapter_norm_v2  32 &
run_model 1 Salesforce/codet5p-770m                codet5p_clone_search_adapter_norm_v2      8 &
run_model 2 deepseek-ai/deepseek-coder-1.3b-base   deepseek_clone_search_adapter_norm_v2     4 &
run_model 3 Qwen/Qwen2.5-Coder-1.5B               qwen25_clone_search_adapter_norm_v2       8 &

wait

echo ""
echo "=== Rerun complete — $(date) ==="
