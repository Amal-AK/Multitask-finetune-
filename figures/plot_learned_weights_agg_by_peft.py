import re
import ast
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS_DIR = "/home/aakli/Multitask-finetune-/logs"

RUNS = [
    ("unix_full_norm_v2_300k.log",              "UniXcoder-base",       "Full"),
    ("unix_adapter_norm_v2_300k.log",           "UniXcoder-base",       "Serial adapter"),
    ("unix_parallel_adapter_norm_v2_300k.log",  "UniXcoder-base",       "Parallel adapter"),
    ("unix_lora_norm_v2.log",                   "UniXcoder-base",       "LoRA"),
    ("unixcoder_mlora_norm_v2_300k.log",        "UniXcoder-base",       "MLoRA"),

    ("modernbert_full_norm_v2_300k.log",             "ModernBERT-base", "Full"),
    ("modernbert_adapter_norm_v2_300k.log",          "ModernBERT-base", "Serial adapter"),
    ("modernbert_parallel_adapter_norm_v2_300k.log", "ModernBERT-base", "Parallel adapter"),
    ("modernbert_lora_norm_v2_300k.log",             "ModernBERT-base", "LoRA"),
    ("modernbert_mlora_norm_v2_300k.log",            "ModernBERT-base", "MLoRA"),

    ("codet5p_full_norm_v2_300k.log",             "CodeT5+ 770M", "Full"),
    ("codet5p_adapter_norm_v2_300k.log",          "CodeT5+ 770M", "Serial adapter"),
    ("codet5p_parallel_adapter_norm_v2_300k.log", "CodeT5+ 770M", "Parallel adapter"),
    ("codet5p_lora_norm_v2_300k.log",             "CodeT5+ 770M", "LoRA"),
    ("codet5p_mlora_norm_v2_300k.log",            "CodeT5+ 770M", "MLoRA"),

    ("deepseek_coder_1.3b_full_norm_v2_300k.log",             "DeepSeek-Coder 1.3B", "Full"),
    ("deepseek_coder_1.3b_adapter_norm_v2_300k.log",          "DeepSeek-Coder 1.3B", "Serial adapter"),
    ("deepseek_coder_1.3b_parallel_adapter_norm_v2_300k.log", "DeepSeek-Coder 1.3B", "Parallel adapter"),
    ("deepseek_coder_1.3b_lora_norm_v2_300k.log",             "DeepSeek-Coder 1.3B", "LoRA"),

    ("qwen25_coder_1.5b_full_norm_v2_300k.log",             "Qwen2.5-Coder-1.5B", "Full"),
    ("qwen25_coder_1.5b_adapter_norm_v2_300k.log",          "Qwen2.5-Coder-1.5B", "Serial adapter"),
    ("qwen25_coder_1.5b_parallel_adapter_norm_v2_300k.log", "Qwen2.5-Coder-1.5B", "Parallel adapter"),
    ("qwen25_coder_1.5b_lora_norm_v2_300k.log",             "Qwen2.5-Coder-1.5B", "LoRA"),
]

TASKS = ["vul_detection", "clone_detection", "code_search", "flakiness_detect"]
TASK_LABEL = {"vul_detection": "Vulnerability Detection", "clone_detection": "Clone Detection",
              "code_search": "Code Search", "flakiness_detect": "Flakiness Detection"}
COLORS = {"vul_detection": "#d62728", "clone_detection": "#1f77b4",
          "code_search": "#2ca02c", "flakiness_detect": "#ff7f0e"}

EPOCH_CAP = 5
models_order = ["UniXcoder-base", "ModernBERT-base", "CodeT5+ 770M", "DeepSeek-Coder 1.3B", "Qwen2.5-Coder-1.5B"]

def extract_weight_trajectory(path):
    text = open(path, errors="ignore").read()
    lines = re.findall(r"Normalized weights → next epoch: (\{.*\})", text)
    traj = [{"vul_detection": 1.0, "clone_detection": 1.0,
             "code_search": 1.0, "flakiness_detect": 1.0}]
    for l in lines:
        d = ast.literal_eval(l)
        traj.append({k: float(v) for k, v in d.items()})
    traj = traj[:-1] if len(traj) > 1 else traj
    return traj[:EPOCH_CAP]

grid = {m: {} for m in models_order}
for fname, model, method in RUNS:
    path = os.path.join(LOGS_DIR, fname)
    if not os.path.exists(path):
        continue
    traj = extract_weight_trajectory(path)
    if len(traj) <= 1:
        continue
    grid[model][method] = traj

fig, axes = plt.subplots(1, len(models_order), figsize=(4.6 * len(models_order), 5.2), sharey=True)

for i, model in enumerate(models_order):
    ax = axes[i]
    methods = list(grid[model].keys())
    n_peft = len(methods)
    for task in TASKS:
        # stack per-epoch values across PEFT methods, padding short runs with NaN
        mat = np.full((n_peft, EPOCH_CAP), np.nan)
        for k, method in enumerate(methods):
            traj = grid[model][method]
            for e, d in enumerate(traj):
                mat[k, e] = d.get(task, np.nan)
        mean = np.nanmean(mat, axis=0)
        vmin = np.nanmin(mat, axis=0)
        vmax = np.nanmax(mat, axis=0)
        epochs = np.arange(1, EPOCH_CAP + 1)
        valid = ~np.isnan(mean)
        ax.plot(epochs[valid], mean[valid], color=COLORS[task], marker="o",
                 markersize=7, linewidth=3.2, label=TASK_LABEL[task])
        ax.fill_between(epochs[valid], vmin[valid], vmax[valid], color=COLORS[task], alpha=0.15)
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xticks(range(1, EPOCH_CAP + 1))
    ax.set_title(model, fontsize=24, pad=14)
    ax.set_xlabel("Epoch", fontsize=21)
    ax.tick_params(labelsize=19)
    ax.grid(alpha=0.3, linewidth=0.6)

axes[0].set_ylabel("Loss weight $a_i$", fontsize=24)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=23, bbox_to_anchor=(0.5, -0.1),
           markerscale=2.0, handlelength=3)
fig.tight_layout()

out_path = "/tmp/claude-1002/-home-aakli-Multitask-finetune-/6f1cf742-3a9b-4bc7-be29-a8bb706f6044/scratchpad/learned_weights_agg_by_peft.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("saved:", out_path)
