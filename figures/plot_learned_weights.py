import re
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS_DIR = "/home/aakli/Multitask-finetune-/logs"

# (log filename, model label, method label)
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

def extract_weight_trajectory(path):
    text = open(path, errors="ignore").read()
    lines = re.findall(r"Normalized weights → next epoch: (\{.*\})", text)
    traj = [{"vul_detection": 1.0, "clone_detection": 1.0,
             "code_search": 1.0, "flakiness_detect": 1.0}]  # epoch 1 = uniform
    for l in lines:
        d = ast.literal_eval(l)
        d = {k: float(v) for k, v in d.items()}
        traj.append(d)
    # last entry is "for next epoch" beyond what was actually used if training
    # stopped early; keep all — they still show the trend.
    return traj[:-1] if len(traj) > 1 else traj

methods_order = ["Full", "Serial adapter", "Parallel adapter", "LoRA", "MLoRA"]
INCOMPLETE_RUNS = {("CodeT5+ 770M", "MLoRA")}  # still training as of last log update
models_order = ["UniXcoder-base", "ModernBERT-base", "CodeT5+ 770M", "DeepSeek-Coder 1.3B", "Qwen2.5-Coder-1.5B"]

grid = {m: {} for m in models_order}
for fname, model, method in RUNS:
    import os
    path = os.path.join(LOGS_DIR, fname)
    if not os.path.exists(path):
        continue
    traj = extract_weight_trajectory(path)
    if len(traj) <= 1:
        continue
    grid[model][method] = traj

# Fix a shared x-axis range across all panels (runs varied from 3 to 9 epochs
# before early stopping). Most runs are 4-5 epochs, so cap at 5: longer runs
# get truncated to 5, shorter ones (only the 3- and 4-epoch runs) simply end
# early rather than forcing everything down to the shortest run.
EPOCH_CAP = 5
for model in grid:
    for method in grid[model]:
        grid[model][method] = grid[model][method][:EPOCH_CAP]
max_len = max(len(traj) for methods in grid.values() for traj in methods.values())

all_vals = [d.get(t, float("nan")) for methods in grid.values() for traj in methods.values()
            for d in traj for t in TASKS]
y_min, y_max = min(all_vals), max(all_vals)
pad = 0.05 * (y_max - y_min)
Y_LIM = (max(0, y_min - pad), y_max + pad)

n_rows = len(models_order)
n_cols = len(methods_order)
# Sized for A3 printing (scaled to fit the page, not forced to the exact
# 16.53x11.69 ratio) — tall enough per row that the bigger fonts below don't collide.
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.3 * n_rows), sharex=True, sharey=True)

for i, model in enumerate(models_order):
    for j, method in enumerate(methods_order):
        ax = axes[i, j]
        traj = grid[model].get(method)
        if traj is None:
            ax.axis("off")
            continue
        epochs = list(range(1, len(traj) + 1))
        for t in TASKS:
            vals = [d.get(t, float("nan")) for d in traj]
            ax.plot(epochs, vals, color=COLORS[t], marker="o", markersize=6,
                    linewidth=3.0, label=TASK_LABEL[t])
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":")
        ax.set_xticks(range(1, max_len + 1))
        ax.set_xlim(1, max_len)
        ax.set_ylim(*Y_LIM)
        ax.tick_params(labelsize=15)
        if i == 0:
            ax.set_title(method, fontsize=20, pad=12)
        if j == 0:
            ax.set_ylabel(model, fontsize=16, labelpad=12)
        if (model, method) in INCOMPLETE_RUNS:
            ax.text(0.98, 0.95, "in progress", transform=ax.transAxes,
                    fontsize=13, style="italic", ha="right", va="top", color="gray")
        ax.grid(alpha=0.3, linewidth=0.6)

handles, labels = None, None
for i in range(n_rows):
    for j in range(n_cols):
        if grid[models_order[i]].get(methods_order[j]) is not None:
            handles, labels = axes[i, j].get_legend_handles_labels()
            break
    if handles:
        break

fig.suptitle("Learned per-task loss weights $a_1,\\dots,a_4$ across training epochs\n(normalized weighting, MFT)",
             fontsize=22, y=1.0)
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=18, bbox_to_anchor=(0.5, -0.02),
           markerscale=1.8, handlelength=3)
fig.tight_layout(rect=[0.015, 0.03, 1, 0.93])

out_path = "/tmp/claude-1002/-home-aakli-Multitask-finetune-/6f1cf742-3a9b-4bc7-be29-a8bb706f6044/scratchpad/learned_weights_grid.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("saved:", out_path)

# print a short coverage report
for model in models_order:
    have = [m for m in methods_order if grid[model].get(m) is not None]
    missing = [m for m in methods_order if grid[model].get(m) is None]
    print(model, "OK:", have, "MISSING:", [m for m in missing if m not in ("MLoRA", "MLoRA*") or (model in ("UniXcoder-base","ModernBERT-base","CodeT5+ 770M"))])
