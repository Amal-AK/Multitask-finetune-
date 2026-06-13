#!/bin/bash
# Smoke test — 100 train samples, 1 epoch.
# Covers all 12 combos: {unixcoder, codet5p-770m} × {adapter, parallel_adapter, prefix} × {normalized, uncertainty}
# UniXcoder (GPU 0) and CodeT5+ 770M (GPU 1) run in parallel; runs are sequential within each GPU.
# Pass/fail tracking uses temp files to work correctly across subshells.
#
# Usage:
#   bash smoke_test.sh               # both GPUs, all 12 combos
#   bash smoke_test.sh 0 unix        # only UniXcoder combos on GPU 0
#   bash smoke_test.sh 1 codet5p     # only CodeT5+ combos on GPU 1
set -euo pipefail

mkdir -p logs

GPU0="${1:-0}"
GPU1="${2:-1}"
FILTER="${3:-all}"   # "unix" | "codet5p" | "all"

BDIM=64
TMPDIR_SMOKE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_SMOKE"' EXIT

run_smoke() {
    local label="$1"; local gpu="$2"; shift 2
    local logfile="logs/smoke_${label}.log"
    printf "  [GPU%s] %-52s ... " "$gpu" "$label"
    if conda run --no-capture-output -n multitask \
            env CUDA_VISIBLE_DEVICES="$gpu" python -u run.py "$@" \
            --tasks vul_detection,clone_detection,code_search,flakiness_detect \
            --max_train_samples 100 \
            --max_eval_samples  100 \
            --num_train_epochs  1 \
            --code_length 512 --nl_length 128 \
            --learning_rate 1e-4 \
            > "$logfile" 2>&1; then
        echo "PASS"
        touch "${TMPDIR_SMOKE}/pass_${label}"
    else
        echo "FAIL  (see $logfile)"
        tail -6 "$logfile"
        touch "${TMPDIR_SMOKE}/fail_${label}"
    fi
}

echo "=== Smoke test — 100 samples — $(date) ==="
echo "    adapter_dim=$BDIM  GPUs: unix→$GPU0  codet5p→$GPU1"
echo ""

# ---- GPU 0: UniXcoder (roberta) ----
run_unix() {
    for peft in adapter parallel_adapter prefix; do
        for loss in normalized uncertainty; do
            local tag
            tag="$([ "$loss" = "normalized" ] && echo "norm_v2" || echo "unc_v2")"
            run_smoke "unix_${peft}_${tag}" "$GPU0" \
                --model_name_or_path microsoft/unixcoder-base \
                --peft_module "$peft" \
                --bottleneck_dim $BDIM \
                --loss_weighting "$loss" \
                --train_batch_size 8 --eval_batch_size 8 \
                --output_model_name "smoke_unix_${peft}_${tag}"
        done
    done
}

# ---- GPU 1: CodeT5+ 770M (t5 encoder) ----
run_codet5p() {
    for peft in adapter parallel_adapter prefix; do
        for loss in normalized uncertainty; do
            local tag
            tag="$([ "$loss" = "normalized" ] && echo "norm_v2" || echo "unc_v2")"
            run_smoke "codet5p_${peft}_${tag}" "$GPU1" \
                --model_name_or_path Salesforce/codet5p-770m \
                --peft_module "$peft" \
                --bottleneck_dim $BDIM \
                --loss_weighting "$loss" \
                --train_batch_size 4 --eval_batch_size 4 \
                --output_model_name "smoke_codet5p_${peft}_${tag}"
        done
    done
}

case "$FILTER" in
    unix)    run_unix ;;
    codet5p) run_codet5p ;;
    *)
        # Reset EXIT trap in each subshell so they don't delete TMPDIR_SMOKE on exit
        # (bash subshells inherit parent traps; premature deletion breaks the counters).
        # Pass/fail is tracked via temp files, so wait's exit code is not meaningful.
        ( trap - EXIT; run_unix    ) &
        ( trap - EXIT; run_codet5p ) &
        wait || true
        ;;
esac

# Count via nullglob arrays — avoids ls exiting non-zero when no fail_* files exist,
# which would trigger set -o pipefail and kill the script before printing Results.
shopt -s nullglob
pass_files=("${TMPDIR_SMOKE}"/pass_*)
fail_files=("${TMPDIR_SMOKE}"/fail_*)
shopt -u nullglob
PASS=${#pass_files[@]}
FAIL=${#fail_files[@]}
echo ""
echo "=== Results: $PASS passed, $FAIL failed — $(date) ==="
[ "$FAIL" -eq 0 ]
