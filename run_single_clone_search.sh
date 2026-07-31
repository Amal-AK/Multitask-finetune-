#!/bin/bash
# Single-task finetune: clone_detection (GPU 2) and code_search (GPU 3) — ModernBERT
# Each GPU: parallel_adapter first, then full finetune, sequentially.
set -euo pipefail

mkdir -p logs/single_task

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64
LOSS="normalized"
MODEL="answerdotai/ModernBERT-base"
BATCH=32

run_job() {
    local GPU="$1" TASK="$2" PEFT="$3" LABEL="$4"
    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
            --model_name_or_path "$MODEL" \
            --peft_module "$PEFT" --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks "$TASK" \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size $BATCH --eval_batch_size $BATCH \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --output_model_name "$LABEL" \
            2>&1 | tee "logs/single_task/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

run_gpu2() {
    run_job 2 clone_detection parallel_adapter modernbert_clone_parallel_adapter_norm_v2
    run_job 2 clone_detection full             modernbert_clone_full_norm_v2
}

run_gpu3() {
    run_job 3 code_search parallel_adapter modernbert_search_parallel_adapter_norm_v2
    run_job 3 code_search full             modernbert_search_full_norm_v2
}

echo "=== Single-task finetune (clone / code_search) — $(date) ==="
echo "    GPU2: clone_detection (parallel_adapter -> full)"
echo "    GPU3: code_search     (parallel_adapter -> full)"
echo ""

run_gpu2 &
run_gpu3 &
wait

echo ""
echo "=== Single-task finetune complete — $(date) ==="
