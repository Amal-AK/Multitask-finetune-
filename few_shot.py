# ───────────────────────────────────────────── main ──────────────────────────────────

import argparse
import logging
import multiprocessing as mp
import os
import random
import re
import signal
import textwrap
import torch, gc
import sys
from typing import Any, Tuple , List, Dict
import ast
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import json, random
# Assuming utilities is a local file you have; if not, ensure the import is valid or removed if unused
# from utilities import TextDataset_code_search 

# ─────────────────────────────── global settings & logging ─────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

transformers.logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger("few_shot_eval")


# ───────────────────────────────── reproducibility helper ──────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 



# ───────────────────────── forced single-token helpers ─────────────────────────

def extract_label(raw_prompt: str,
                     allowed: List[str],
                     generator,
                     model_name: str,
                     tokenizer,
                     hf_max_new_tokens: int = 10) -> str:
    """
    Ask the model but hard-limit to tokens to determine 0 or 1.
    """
    
    prompt = raw_prompt
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    
    # We allow slightly more tokens to catch "The answer is No"
    out = generator(
            formatted,
            max_new_tokens=hf_max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"].strip()

    # Simple heuristic to map text to 0/1
    # 0 = Negative (No, Safe, Non-Vulnerable, Non-Flaky)
    # 1 = Positive (Yes, Vulnerable, Flaky, Clone)
    
    out_lower = out.lower()
    if "not" in out_lower or "no" in out_lower or "safe" in out_lower or "clean" in out_lower:
        y = '0'
    else:
        # Default to 1 if it says "yes", "vulnerable", or ambiguous
        y = '1'
        
    if y not in set(allowed):
        # Fallback if strict validation is needed, but for binary we usually clamp
        return '0' 
        
    return y


def ask_discrete_score(raw_prompt: str, valid_digits: List[str], generator, model_name: str, tokenizer) -> int:
    prompt = raw_prompt
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    out = generator(formatted, max_new_tokens=5, do_sample=False, return_full_text=False)[0]["generated_text"].strip()
    m = re.search(r"\b([0-3])\b", out)
    return int(m.group(1)) if m else 0


# ─────────────────────────────── general evaluators ───────────────────────────────

def eval_binary_task(args,
                     generator,
                     tokenizer,
                     rows: List[Dict[str, Any]],
                     build_prompt_fn,
                     gold_key: str = "label",
                     allowed_labels: List[str] = ["0", "1"],
                     task_name: str = "binary") -> Dict[str, Any]:
    """
    rows: list of dicts.
    build_prompt_fn(row) -> str
    """
    gold, pred = [], []
    
    # Iterate with progress bar
    for row in tqdm(rows, desc=f"{task_name}: samples", unit="ex", dynamic_ncols=True, miniters=100):
        raw_prompt = build_prompt_fn(row)
        y = extract_label(raw_prompt, allowed_labels, generator, args.modelName, tokenizer)
        
        pred.append(y)
        gold_val = str(int(row[gold_key])) if isinstance(row[gold_key], (int, bool)) else str(row[gold_key])
        gold.append(gold_val)
        
        
    labels_sorted = sorted(allowed_labels)
    acc = accuracy_score(gold, pred)
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, labels=labels_sorted, average="macro", zero_division=0)
    mcc = matthews_corrcoef(gold, pred)

    # per-class breakdown
    per_label = {}
    prfs = precision_recall_fscore_support(gold, pred, labels=labels_sorted, zero_division=0)
    for i, lbl in enumerate(labels_sorted):
        per_label[lbl] = {"precision": round(prfs[0][i], 4),
                          "recall":    round(prfs[1][i], 4),
                          "f1":        round(prfs[2][i], 4)}
    
    per_label_mcc = {}
    for lbl in labels_sorted:
        gb = [1 if g == lbl else 0 for g in gold]
        pb = [1 if p_ == lbl else 0 for p_ in pred]
        per_label_mcc[lbl] = round(matthews_corrcoef(gb, pb), 4)

    return {
        "Model":       args.modelName,
        "Task":        task_name,
        "Samples":     len(gold),
        "Accuracy":    round(acc, 4),
        "Macro_P":     round(p, 4),
        "Macro_R":     round(r, 4),
        "Macro_F1":    round(f1, 4),
        "MCC":         round(mcc, 4),
        "PerLabelMCC": per_label_mcc,
        "PerLabel":    per_label,
    }


def eval_codesearch_task(args, model, tokenizer, codesearch_file: str, k: int, task_name: str = "code_search") -> Dict[str, Any]:
    """
    CodeSearch eval (Dot product retrieval + LLM Re-ranking).
    This remains mostly unchanged as it is not a simple binary classification task.
    """
    import os, json, tempfile, heapq, re
    import numpy as np, torch
    
    # ---------- load minimal fields ----------
    code_texts, nl_texts, urls = [], [], []
    with open(codesearch_file, "r", encoding="utf-8") as f:
        for line in f:
            js = json.loads(line)
            # Handle variable field names
            if isinstance(js.get("function_tokens"), list): code = " ".join(js["function_tokens"])
            elif isinstance(js.get("code_tokens"), list): code = " ".join(js["code_tokens"])
            else: code = " ".join(str(js.get("function", js.get("code", ""))).split())
            
            if isinstance(js.get("docstring_tokens"), list): nl = " ".join(js["docstring_tokens"])
            else: nl = " ".join(str(js.get("docstring", js.get("doc", ""))).split())
            
            url = js.get("url", js.get("retrieval_idx"))
            if url is None: continue
            code_texts.append(code); nl_texts.append(nl); urls.append(url)

    N = len(code_texts)
    if N == 0: return {"Task": task_name, "Model": args.modelName, "Queries": 0, "K": k, "MRR@K": 0.0}

    bs = getattr(args, "eval_batch_size", 4)
    code_len = getattr(args, "code_length", 256)
    nl_len   = getattr(args, "nl_length", 128)
    cutoff   = min(k, N) if k else N

    # Re-ranking params
    topM = min(100, cutoff)
    cands_per_prompt = 20
    max_code_chars = 800

    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = getattr(model, "model", None) or getattr(model, "transformer", None) or model
    model.config.use_cache = False
    model.eval(); base.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s n_gpu=%s", device, torch.cuda.device_count())
    hidden_size = getattr(getattr(base, "config", model.config), "hidden_size", None)
    
    # ---------- helper: embed batch ----------
    def embed_batch(text_batch: List[str], max_len: int) -> np.ndarray:
        enc = tokenizer(text_batch, padding="max_length", truncation=True,
                        max_length=max_len, add_special_tokens=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=getattr(args, "autocast_dtype", torch.float16), enabled=input_ids.is_cuda):
                out = base(input_ids=input_ids, attention_mask=attn_mask, return_dict=True, use_cache=False)
                last = out.last_hidden_state
                masked = last * attn_mask.unsqueeze(-1)
                lens = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
                vec = (masked.sum(dim=1) / lens).float()
        vec_cpu = vec.cpu().numpy().astype(np.float16)
        del enc, input_ids, attn_mask, out, last, masked, lens, vec
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return vec_cpu

    # ---------- 1) Build code embeddings ----------
    mmap_path = os.path.join(tempfile.gettempdir(), f"code_vecs_{os.getpid()}.dat")
    code_vecs = np.memmap(mmap_path, mode="w+", dtype=np.float16, shape=(N, hidden_size))
    for start in tqdm(range(0, N, bs), desc=f"{task_name}: embed code", unit="batch", dynamic_ncols=True):
        end = min(start + bs, N)
        code_vecs[start:end] = embed_batch(code_texts[start:end], code_len)

    # ---------- Retrieval Helpers ----------
    def topk_by_dot(q: np.ndarray, K: int, chunk: int = 2048) -> List[int]:
        heap = []
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            block = code_vecs[c0:c1].astype(np.float32)
            scores = block @ q
            if K < len(scores): part = np.argpartition(scores, -K)[-K:]
            else: part = np.arange(len(scores))
            for off in part:
                s = float(scores[off]); idx = c0 + int(off)
                if len(heap) < K: heapq.heappush(heap, (s, idx))
                elif s > heap[0][0]: heapq.heapreplace(heap, (s, idx))
        heap.sort(key=lambda x: x[0], reverse=True)
        return [idx for s, idx in heap]

    def build_rerank_prompt(query: str, ids: List[int], codes: List[str]) -> str:
        head = (
            "Act as a code-search re-ranker. For the QUERY and CANDIDATES, "
            "score each candidate 0–100 (100 = exact implementation, 0 = unrelated). "
            "Output one line per candidate: <ID>\\t<score>. No extra text.\n\n"
            f"QUERY:\n---\n{query}\n---\n\nCANDIDATES:\n"
        )
        body = []
        for i, code in zip(ids, codes):
            c = code[:max_code_chars]
            body.append(f"ID={i}\nCODE:\n---\n{c}\n---\n")
        return head + "\n".join(body)

    score_line = re.compile(r"^\s*(\d+)\s*[\t ,:|-]\s*(\d+(?:\.\d+)?)\s*$")

    @torch.inference_mode()
    def ask_scores(prompt: str) -> Dict[int, float]:
        formatted = build_chat_prompt(prompt, args.modelName, tokenizer)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=1024, do_sample=False, temperature=0.0, top_p=1.0,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores: Dict[int, float] = {}
        for line in text.strip().splitlines():
            m = score_line.match(line)
            if m:
                idx = int(m.group(1)); s = float(m.group(2))
                scores[idx] = max(0.0, min(100.0, s))
        return scores

    # ---------- 2) Stream queries ----------
    ranks = []
    chunk = max(2048 // max(hidden_size // 1024, 1), 256)

    for start in tqdm(range(0, N, bs), desc=f"{task_name}: rerank", unit="batch", dynamic_ncols=True):
        end = min(start + bs, N)
        q_vecs = embed_batch(nl_texts[start:end], nl_len).astype(np.float32)

        for j in range(end - start):
            gold_idx = start + j
            q = q_vecs[j]
            topk_idx = topk_by_dot(q, cutoff, chunk=chunk)

            if gold_idx not in topk_idx:
                ranks.append(0.0)
                continue

            rerank_ids = topk_idx[:topM]
            llm_scores: Dict[int, float] = {}
            for b0 in range(0, len(rerank_ids), cands_per_prompt):
                b1 = min(b0 + cands_per_prompt, len(rerank_ids))
                ids_batch = rerank_ids[b0:b1]
                codes_batch = [code_texts[t] for t in ids_batch]
                prompt = build_rerank_prompt(nl_texts[gold_idx], ids_batch, codes_batch)
                llm_scores.update(ask_scores(prompt))

            scored = [idx for idx in rerank_ids if idx in llm_scores]
            unscored = [idx for idx in rerank_ids if idx not in llm_scores]
            scored.sort(key=lambda x: llm_scores[x], reverse=True)
            final_order = scored + unscored + [idx for idx in topk_idx if idx not in rerank_ids]

            rank = final_order.index(gold_idx) + 1
            ranks.append(1.0 / rank)

    mrr = float(np.mean(ranks)) if ranks else 0.0
    try:
        del code_vecs
        os.remove(mmap_path)
    except: pass

    return {"Task": task_name, "Model": args.modelName, "Queries": len(ranks), "K": cutoff, "MRR@K": round(mrr, 4)}


# ─────────────────────────────── task loaders + prompts ───────────────────────────────

def _sample_few_shot_block(pos_pool, neg_pool, current_code_signature, n_pos=2, n_neg=2, format_type="single"):
    """
    Helper to sample examples and format them.
    format_type: 'single' for Vuln/Flaky, 'pair' for Clone.
    """
    # Exclude current to avoid data leakage
    p_cands = [p for p in pos_pool if p["id"] != current_code_signature]
    n_cands = [n for n in neg_pool if n["id"] != current_code_signature]
    
    # Sample
    curr_pos = random.sample(p_cands, min(len(p_cands), n_pos))
    curr_neg = random.sample(n_cands, min(len(n_cands), n_neg))
    
    block = ""
    
    if format_type == "single":
        # Add Positives
        for item in curr_pos:
            block += f"EXAMPLE (POSITIVE/YES):\nCode:\n{item['code'][:500]}\nLABEL: YES\n-----\n"
        # Add Negatives
        for item in curr_neg:
            block += f"EXAMPLE (NEGATIVE/NO):\nCode:\n{item['code'][:500]}\nLABEL: NO\n-----\n"
            
    elif format_type == "pair":
        # Add Positives
        for item in curr_pos:
            block += (f"EXAMPLE (CLONE/YES):\nSnippet A:\n{item['code1'][:300]}\n"
                      f"Snippet B:\n{item['code2'][:300]}\nLABEL: YES\n-----\n")
        # Add Negatives
        for item in curr_neg:
            block += (f"EXAMPLE (NON-CLONE/NO):\nSnippet A:\n{item['code1'][:300]}\n"
                      f"Snippet B:\n{item['code2'][:300]}\nLABEL: NO\n-----\n")
            
    return block


# ─── 1. Vulnerability ───

def load_vuln_rows(path: str) -> List[Dict[str, Any]]:
    # Load all to memory
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            code_clean = ' '.join(r["func"].split())
            rows.append({
                "code": code_clean, 
                "label": int(r["target"]),
                "id": hash(code_clean) # Simple hash for exclusion check
            })
            
    # Split pools
    pos_pool = [r for r in rows if r["label"] == 1]
    neg_pool = [r for r in rows if r["label"] == 0]
    
    # Inject examples
    for r in rows:
        r["few_shot"] = _sample_few_shot_block(pos_pool, neg_pool, r["id"], format_type="single")
        
    return rows

def vuln_prompt(row: Dict[str, Any]) -> str:
    return (
        "I want you to act as a vulnerability detection system.\n"
        "Below are examples of VULNERABLE (YES) and NON-VULNERABLE (NO) code.\n"
        "Then I will give you a TARGET code to classify.\n\n"
        "### EXAMPLES:\n" + row["few_shot"] + "\n"
        "### TARGET TASK:\n"
        "Is this code vulnerable?\n"
        "CODE:\n-----\n" + row["code"] + "\n-----\n"
        "Respond with yes or no, NO extra text."
    )


# ─── 2. Flaky Tests ───

def load_flaky_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rows = []
    for r in data:
        code_clean = ' '.join(r["code"].split())
        rows.append({
            "code": code_clean,
            "label": int(r["label"]),
            "id": hash(code_clean)
        })
        
    pos_pool = [r for r in rows if r["label"] == 1]
    neg_pool = [r for r in rows if r["label"] == 0]

    for r in rows:
        r["few_shot"] = _sample_few_shot_block(pos_pool, neg_pool, r["id"], format_type="single")
        
    return rows

def flaky_prompt(row: Dict[str, Any]) -> str:
    return (
        "I want you to act as a test flakiness detection system.\n"
        "Below are examples of FLAKY (YES) and NON-FLAKY (NO) tests.\n\n"
        "### EXAMPLES:\n" + row["few_shot"] + "\n"
        "### TARGET TASK:\n"
        "Is this TEST FLAKY?\n"
        "TEST CODE:\n-----\n" + row["code"] + "\n-----\n"
        "Respond with yes or no, NO extra text."
    )


# ─── 3. Clone Detection ───

def load_clone_rows_txt(code_file: str, pairs_txt: str , limit: int = None) -> List[Dict[str, Any]]:
    # 1. Load Code Map
    idx2code = {}
    with open(code_file, "r", encoding="utf-8") as f:
        for ln in f:
            js = json.loads(ln)
            idx2code[js["idx"]] = ' '.join(js["func"].split())

    # 2. Load Pairs
    rows = []
    with open(pairs_txt, "r", encoding="utf-8") as f:
        for ln in f:
            parts = re.split(r"\s+", ln.strip())
            if len(parts) != 3: continue
            a, b, lab = parts
            if a in idx2code and b in idx2code:
                # We use a combined string as ID to avoid self-sampling
                combo_id = f"{a}_{b}"
                rows.append({
                    "code1": idx2code[a],
                    "code2": idx2code[b],
                    "label": int(lab),
                    "id": combo_id 
                })
            # NOTE: We load ALL first to build the few-shot pool, then we slice for the limit later
            
    # 3. Create Pools from valid data
    pos_pool = [r for r in rows if r["label"] == 1]
    neg_pool = [r for r in rows if r["label"] == 0]
    
    # 4. Apply Limit (if any) to the rows we want to EVALUATE, 
    #    but keep pools large for sampling.
    if limit is not None:
        eval_rows = rows[:limit]
    else:
        eval_rows = rows

    # 5. Inject Examples
    for r in eval_rows:
        r["few_shot"] = _sample_few_shot_block(pos_pool, neg_pool, r["id"], format_type="pair")
            
    return eval_rows

def clone_prompt(row: Dict[str, Any]) -> str:
    return (
        "I want you to act as a code clone detection system.\n"
        "Below are examples of SEMANTIC CLONES (YES) and NON-CLONES (NO).\n\n"
        "### EXAMPLES:\n" + row["few_shot"] + "\n"
        "### TARGET TASK:\n"
        "Are the two snippets SEMANTIC CLONES?\n"
        "SNIPPET A:\n-----\n" + row["code1"] + "\n-----\n\n"
        "SNIPPET B:\n-----\n" + row["code2"] + "\n-----\n"
        "Respond with yes or no, NO extra text."
    )


# ─────────────────────────────── Chat Template Logic ───────────────────────────────

def build_chat_prompt(prompt: str, model_name: str, tokenizer=None) -> str:
    # 1. Tokenizer native
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    name = model_name.lower()
    # 2. Manual fallbacks
    if "llama" in name or "mistral" in name or "zephyr" in name:
        return f"<s>[INST] {prompt.strip()} [/INST]"
    if "deepseek" in name:
        return f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"
    if "starcoder" in name or "santacoder" in name:
        return prompt.strip() + (tokenizer.eos_token if tokenizer else "")
    if "qwen" in name:
        return f"<|im_start|>user\n{prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    return prompt.strip()


def pretty_print_summary(task_summaries: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(task_summaries)
    col_order = [c for c in ["Task","Model","Samples","Accuracy","Macro_F1","MCC","Queries","K","MRR@K"] if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]

    for col in ["Accuracy","Macro_P","Macro_R","Macro_F1","MCC","MRR@K"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    print("\n" + "="*68)
    print("EVALUATION SUMMARY".center(68))
    print("="*68)
    print(df.to_string(index=False))
    print("="*68 + "\n")


# ─────────────────────────────── Main Execution ───────────────────────────────

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate LLM code generation on Binary Tasks + CodeSearch")
    parser.add_argument("--modelName", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF model id")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    # Task toggles
    parser.add_argument("--run_vuln" , action="store_true")
    parser.add_argument("--vuln_file", type=str, default="./datasets/dataset_vulnerabilty/test.jsonl")

    parser.add_argument("--run_flaky", action="store_true")
    parser.add_argument("--flaky_file", type=str, default="./datasets/dataset_flakytest/valid.json")

    parser.add_argument("--run_clone", action="store_true")
    parser.add_argument("--clone_code_file", type=str, default="./datasets/dataset_clone/data.jsonl")
    parser.add_argument("--clone_pairs_file", type=str, default="./datasets/dataset_clone/test.txt")

    parser.add_argument("--run_codesearch", action="store_true")
    parser.add_argument("--codesearch_file", type=str, default="./datasets/code_search/test.jsonl")
    parser.add_argument("--cs_topk", type=int, default=1000)
    
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--autocast_dtype", default=torch.float16, type=lambda _: torch.float16)

    args = parser.parse_args()
    set_seed(args.seed)

    Models = [  
        "Qwen/Qwen2.5-32B-Instruct",
        #"codellama/CodeLlama-34b-Instruct-hf",
        # Add your other models here
    ]
    
    for model_name in Models:
        args.modelName = model_name
        logger.info(f"Loading Model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            do_sample=False,
        )
        
        task_summaries = []

        # 1. Vulnerability
        if args.run_vuln:
            vuln_rows = load_vuln_rows(args.vuln_file)
            logger.info("running model %s on %d vulnerability samples (Few-Shot)", args.modelName, len(vuln_rows))
            s = eval_binary_task(args, generator, tokenizer, vuln_rows, vuln_prompt, 
                                 gold_key="label", allowed_labels=["0","1"], 
                                 task_name="vulnerability_detection")
            task_summaries.append(s)
            pretty_print_summary(task_summaries)
            torch.cuda.empty_cache()

        # 2. Flaky Tests
        if args.run_flaky:
            flaky_rows = load_flaky_rows(args.flaky_file)
            logger.info("running model %s on %d flaky samples (Few-Shot)", args.modelName, len(flaky_rows))
            s = eval_binary_task(args, generator, tokenizer, flaky_rows, flaky_prompt, 
                                 gold_key="label", allowed_labels=["0","1"], 
                                 task_name="flaky_tests")
            task_summaries.append(s)
            pretty_print_summary(task_summaries)    
            torch.cuda.empty_cache()
        
        # 3. Clone Detection
        if args.run_clone:
            clone_rows = load_clone_rows_txt(args.clone_code_file, args.clone_pairs_file , limit=30000)  
            logger.info("running model %s on %d clone samples (Few-Shot)", args.modelName, len(clone_rows))
            s = eval_binary_task(args, generator, tokenizer, clone_rows, clone_prompt,
                                 gold_key="label", allowed_labels=["0","1"],
                                 task_name="clone_detection")
            task_summaries.append(s)
            pretty_print_summary(task_summaries)      
            torch.cuda.empty_cache()
        
        # 4. Code Search
        if args.run_codesearch:
            logger.info("running model %s on code search file: %s", args.modelName, args.codesearch_file)
            s = eval_codesearch_task(args, model, tokenizer, args.codesearch_file, k=args.cs_topk, task_name="code_search")
            task_summaries.append(s)
            pretty_print_summary(task_summaries)

        del model, tokenizer, generator
        gc.collect()
        torch.cuda.empty_cache()