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
from utilities import TextDataset_code_search

# ─────────────────────────────── global settings & logging ─────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

transformers.logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,   # << send logs to STDOUT, not STDERR
    force=True           # overwrite any earlier config
)

logger = logging.getLogger("zero_shot")


# ───────────────────────────────── reproducibility helper ──────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]



# ───────────────────────── forced single-token helpers ─────────────────────────

def extract_label(raw_prompt: str,
                     allowed: List[str],
                     generator,
                     model_name: str,
                     tokenizer,
                     hf_max_new_tokens: int = 10) -> str:
    """
    Ask the model but hard-limit to one token from `allowed`.
    Raises ValueError if the output is not in the allowed set.
    """
    
    
    prompt = raw_prompt
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    out = generator(
            formatted,
            max_new_tokens=hf_max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"].strip()

    y = '0' if "not" in out.lower() or "no" in out.lower() else '1' 
    if y not in set(allowed):
        raise ValueError(f"Invalid label {y!r}. Allowed: {allowed}")
    return y




def ask_discrete_score(raw_prompt: str, valid_digits: List[str], generator, model_name: str, tokenizer) -> int:
    prompt = raw_prompt
    
    formatted = build_chat_prompt(prompt, model_name, tokenizer)
    out = generator(formatted, max_new_tokens=2, do_sample=False, return_full_text=False)[0]["generated_text"].strip()

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
    rows: list of dicts with at least `gold_key`.
    build_prompt_fn(row) -> str
    """
    gold, pred = [], []
    
    for row in tqdm(rows, desc=f"{task_name}: samples", unit="ex", dynamic_ncols=True , miniters=100):  # <- ajouté
        raw_prompt = build_prompt_fn(row)
        y = extract_label(raw_prompt, allowed_labels, generator, args.modelName, tokenizer)
        pred.append(y)
        gold.append(str(int(row[gold_key])) if isinstance(row[gold_key], (int, bool)) else str(row[gold_key]))
        
        
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
    # per-class MCC
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






def eval_codesearch_task(args,
                         model,
                         tokenizer,
                         codesearch_file: str,
                         k: int,
                         task_name: str = "code_search") -> Dict[str, Any]:
    """
    CodeSearch eval with first-stage dot-product retrieval and LLM re-ranking.
    - Builds code embeddings once (memmap + chunks)
    - For each NL query: retrieve top-K by dot-product, then ask the LLM to score top-M
      candidates with a strict prompt (<ID>\t<score> in [0,100]).
    - Final rank = [LLM-scored top-M desc] + [rest of top-K in original order].
    - Reports MRR@K (comparable to fine-tuned baseline).
    """
    import os, json, tempfile, heapq, re
    from typing import List, Dict, Any
    import numpy as np, torch
    from tqdm import tqdm

    # ---------- load minimal fields (strings only) ----------
    code_texts, nl_texts, urls = [], [], []
    with open(codesearch_file, "r", encoding="utf-8") as f:
        for line in f:
            js = json.loads(line)
            if isinstance(js.get("function_tokens"), list):
                code = " ".join(js["function_tokens"])
            elif isinstance(js.get("code_tokens"), list):
                code = " ".join(js["code_tokens"])
            else:
                code = " ".join(str(js.get("function", js.get("code", ""))).split())
            if isinstance(js.get("docstring_tokens"), list):
                nl = " ".join(js["docstring_tokens"])
            else:
                nl = " ".join(str(js.get("docstring", js.get("doc", ""))).split())
            url = js.get("url", js.get("retrieval_idx"))
            if url is None:
                continue
            code_texts.append(code); nl_texts.append(nl); urls.append(url)

    N = len(code_texts)
    if N == 0:
        return {"Task": task_name, "Model": args.modelName, "Queries": 0, "K": k, "MRR@K": 0.0}

    bs = getattr(args, "eval_batch_size", 4)
    code_len = getattr(args, "code_length", 256)
    nl_len   = getattr(args, "nl_length", 128)
    cutoff   = min(k, N) if k else N

    # second-stage params (local defaults; no CLI changes)
    topM = min(100, cutoff)         # how many of top-K to re-rank with the LLM
    cands_per_prompt = 20           # candidates per LLM call (fit context window)
    max_code_chars = 800            # truncate long code blocks in prompts

    # ensure padding token
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # base (encoder) part of the causal LM (LLaMA: .model, Qwen: .transformer)
    base = getattr(model, "model", None) or getattr(model, "transformer", None) or model
    model.config.use_cache = False
    model.eval(); base.eval()

    device = next(model.parameters()).device
    hidden_size = getattr(getattr(base, "config", model.config), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Cannot infer hidden_size from model config")

    # ---------- helper: embed a list of texts ----------
    def embed_batch(text_batch: List[str], max_len: int) -> np.ndarray:
        enc = tokenizer(text_batch, padding="max_length", truncation=True,
                        max_length=max_len, add_special_tokens=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=getattr(args, "autocast_dtype", torch.float16),
                                         enabled=input_ids.is_cuda):
                out = base(input_ids=input_ids, attention_mask=attn_mask,
                           return_dict=True, use_cache=False)
                last = out.last_hidden_state                    # [B, T, H]
                masked = last * attn_mask.unsqueeze(-1)         # mask pads
                lens = attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
                vec = (masked.sum(dim=1) / lens).float()        # mean-pool
        vec_cpu = vec.cpu().numpy().astype(np.float16)          # save memory
        del enc, input_ids, attn_mask, out, last, masked, lens, vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return vec_cpu

    # ---------- 1) build code embeddings to memmap ----------
    mmap_path = os.path.join(tempfile.gettempdir(), f"code_vecs_{os.getpid()}.dat")
    code_vecs = np.memmap(mmap_path, mode="w+", dtype=np.float16, shape=(N, hidden_size))
    for start in tqdm(range(0, N, bs), desc=f"{task_name}: embed code", unit="batch", dynamic_ncols=True):
        end = min(start + bs, N)
        code_vecs[start:end] = embed_batch(code_texts[start:end], code_len)

    # ---------- helpers: top-K retrieval & LLM scoring ----------
    def topk_by_dot(q: np.ndarray, K: int, chunk: int = 2048) -> List[int]:
        """Streaming top-K indices by dot-product vs memmapped code_vecs."""
        heap = []  # min-heap of (score, idx)
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            block = code_vecs[c0:c1].astype(np.float32)     # [C,H]
            scores = block @ q                               # [C]
            if K < len(scores):
                part = np.argpartition(scores, -K)[-K:]
            else:
                part = np.arange(len(scores))
            for off in part:
                s = float(scores[off]); idx = c0 + int(off)
                if len(heap) < K: heapq.heappush(heap, (s, idx))
                elif s > heap[0][0]: heapq.heapreplace(heap, (s, idx))
        heap.sort(key=lambda x: x[0], reverse=True)
        return [idx for s, idx in heap]

    def build_prompt(query: str, ids: List[int], codes: List[str]) -> str:
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
        # Use the same chat templating helper as the rest of the script
        formatted = build_chat_prompt(prompt, args.modelName, tokenizer)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False, temperature=0.0, top_p=1.0,
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

    # ---------- 2) stream queries: retrieve top-K, LLM re-rank top-M, compute rank ----------
    ranks = []
    chunk = max(2048 // max(hidden_size // 1024, 1), 256)  # adaptive chunk to limit RAM

    for start in tqdm(range(0, N, bs), desc=f"{task_name}: rerank", unit="batch", dynamic_ncols=True):
        end = min(start + bs, N)
        q_vecs = embed_batch(nl_texts[start:end], nl_len).astype(np.float32)  # [b, H]

        for j in range(end - start):
            gold_idx = start + j
            q = q_vecs[j]

            # first-stage top-K retrieval
            topk_idx = topk_by_dot(q, cutoff, chunk=chunk)

            # if gold not retrieved in top-K, contribution is 0
            if gold_idx not in topk_idx:
                ranks.append(0.0)
                continue

            # second-stage: LLM scores for top-M
            rerank_ids = topk_idx[:topM]
            llm_scores: Dict[int, float] = {}
            for b0 in range(0, len(rerank_ids), cands_per_prompt):
                b1 = min(b0 + cands_per_prompt, len(rerank_ids))
                ids_batch = rerank_ids[b0:b1]
                codes_batch = [code_texts[t] for t in ids_batch]
                prompt = build_prompt(nl_texts[gold_idx], ids_batch, codes_batch)
                llm_scores.update(ask_scores(prompt))

            # reorder: scored first (desc), then any unscored, then the rest of top-K
            scored = [idx for idx in rerank_ids if idx in llm_scores]
            unscored = [idx for idx in rerank_ids if idx not in llm_scores]
            scored.sort(key=lambda x: llm_scores[x], reverse=True)
            reranked = scored + unscored
            final_order = reranked + [idx for idx in topk_idx if idx not in reranked]

            # compute reciprocal rank of the gold code
            rank = final_order.index(gold_idx) + 1
            ranks.append(1.0 / rank)

    # ---------- 3) summarize ----------
    mrr = float(np.mean(ranks)) if ranks else 0.0
    try:
        del code_vecs
        os.remove(mmap_path)
    except Exception:
        pass

    return {
        "Task":    task_name,
        "Model":   args.modelName,
        "Queries": len(ranks),
        "K":       cutoff,
        "MRR@K":   round(mrr, 4),
    }


# ─────────────────────────────── task loaders + prompts ───────────────────────────────

def load_vuln_rows(path: str) -> List[Dict[str, Any]]:
    # JSONL with {"func": "...", "target": 0/1}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            rows.append({"code": ' '.join(r["func"].split()), "label": int(r["target"])})
    return rows

def vuln_prompt(row: Dict[str, Any]) -> str:
    return (
                    "I want you to act as a vulnerability detection system\n"
                    "Is this code vulnerable?\n"
                    "CODE:\n-----\n" + row["code"] + "\n-----"
                    "Respond with yes or no , NO extra text.\n\n"
                
    )


def load_flaky_rows(path: str) -> List[Dict[str, Any]]:
    # JSON array with {"code": "...", "label": 0/1}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [{"code": ' '.join(r["code"].split()), "label": int(r["label"])} for r in data]
    return rows

def flaky_prompt(row: Dict[str, Any]) -> str:
    return (
        "I want you to act as a test flakiness detection system.\n"
        "Is this TEST FLAKY?\n"
        "TEST CODE:\n-----\n" + row["code"] + "\n-----"
        "Respond with yes or no , NO extra text.\n\n"
    )
    
    
    
def load_clone_rows_txt(code_file: str, pairs_txt: str , limit: int = None) -> List[Dict[str, Any]]:
    """
    code_file: JSONL with {"idx": "...", "func": "..."}
    pairs_txt: plain text file with "idx1<TAB>idx2<TAB>label"
    """
    # First load index→code map
    idx2code = {}
    with open(code_file, "r", encoding="utf-8") as f:
        for ln in f:
            js = json.loads(ln)
            idx2code[js["idx"]] = ' '.join(js["func"].split())

    rows = []
    with open(pairs_txt, "r", encoding="utf-8") as f:
        for ln in f:
            parts = re.split(r"\s+", ln.strip())
            if len(parts) != 3:
                continue
            a, b, lab = parts
            if a in idx2code and b in idx2code:
                rows.append({
                    "code1": idx2code[a],
                    "code2": idx2code[b],
                    "label": int(lab)
                })
            if limit is not None and len(rows) >= limit:
                break
            
    return rows

def clone_prompt(row: Dict[str, Any]) -> str:
    return (
        "I want you to act as a code clone detection system.\n"
        "Are the two snippets SEMANTIC CLONES?\n"
        "SNIPPET A:\n-----\n" + row["code1"] + "\n-----\n\n"
        "SNIPPET B:\n-----\n" + row["code2"] + "\n-----"
        "Respond with yes or no , NO extra text.\n\n"
    )

def load_codesearch_queries(path: str) -> List[Dict[str, Any]]:
    # JSONL with {"query": "...", "candidates": [{"code": "...", "is_positive": true/false}, ...]}
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            qs.append(json.loads(ln))
    return qs


"""
def codesearch_prompt(query_text: str, code_text: str) -> str:
    return (
        "Rate how well the CODE satisfies the QUERY on a discrete scale 0,1,2,3.\n"
        "Return only one digit.\n\n"
        "QUERY:\n-----\n" + query_text + "\n-----\n\n"
        "CODE:\n-----\n" + code_text + "\n-----"
    )

"""




def build_chat_prompt(prompt: str, model_name: str, tokenizer=None) -> str:
    """
    Return a correctly formatted *chat* prompt for almost any HF-style model.
    1. If the `tokenizer` exposes `.apply_chat_template(...)`, use that.
    2. Otherwise fall back to a small set of hard-coded templates.
    3. Fallback-to-fallback: return the raw prompt.
    """
    # -------- 1 : let the tokenizer do the job whenever it can ------------
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            # the new HF API (> 4.38) – safest option
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # some models ship an old or buggy template, drop to rule-table
            pass

    # -------- 2 : minimal rule-table for popular models -------------------
    name = model_name.lower()

    # llama-family (CodeLlama, Llama-2-chat, etc.)
    if "llama" in name:
        return f"<s>[INST] {prompt.strip()} [/INST]"

    # DeepSeek-Coder Instruct
    if "deepseek" in name:
        return f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"

    # StarCoder-family (BigCode, SantaCoder) – they expect plain text + <|endoftext|>
    if "starcoder" in name or "santacoder" in name:
        return prompt.strip() + tokenizer.eos_token

    # Qwen chat / code
    if "qwen" in name:
        return f"<|im_start|>user\n{prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    # Mistral-Instruct / Zephyr / Phi-2-chat – use the same style as Llama-chat
    if any(k in name for k in ("mistral", "zephyr", "phi")):
        return f"<s>[INST] {prompt.strip()} [/INST]"

    # OpenAI ChatCompletion models (if you ever wrap them here)
    if "gpt-3" in name or "gpt-4" in name:

        return prompt.strip()

    # -------- 3 : last resort --------------------------------------------
    return prompt.strip()




def pretty_print_summary(task_summaries: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(task_summaries)

    # Put the important stuff first if present
    col_order = [c for c in ["Task","Model","Samples","Accuracy","Macro_F1","MCC","Queries","K","MRR@K"] if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]

    # Round common metric columns if they exist
    for col in ["Accuracy","Macro_P","Macro_R","Macro_F1","MCC","MRR@K"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    print("\n" + "="*68)
    print("EVALUATION SUMMARY".center(68))
    print("="*68)
    print(df.to_string(index=False))
    print("="*68 + "\n")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate LLM code generation on MBPP / HumanEval")
    parser.add_argument("--modelName", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF model id")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
        # ───────── task toggles and paths ─────────
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s n_gpu=%s", device, torch.cuda.device_count())

    # ─────────── iterate over multiple models ─────────────
    
    
    Models = [  "Qwen/Qwen2.5-Coder-32B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "bigcode/starcoder2-15b",
                "deepseek-ai/deepseek-coder-33b-instruct", 
                "codellama/CodeLlama-34b-Instruct-hf",
                "mistralai/Mistral-7B-Instruct-v0.3",]
    
    
    
        
    for model_name in Models:
            args.modelName = model_name
            # Set the model and tokenizer for the current iteration
    
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",   # force all on GPU
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
            
                    # ───────────────────────── run the four tasks ─────────────────────────
            task_summaries = []

            if args.run_vuln:
                vuln_rows = load_vuln_rows(args.vuln_file)
                logger.info("running model %s on %d vulnerability samples", args.modelName, len(vuln_rows))
                s = eval_binary_task(args, generator, tokenizer, vuln_rows, vuln_prompt, gold_key="label",
                                    allowed_labels=["0","1"], task_name="vulnerability_detection")
                task_summaries.append(s)
            
            if task_summaries:
                pretty_print_summary(task_summaries)
            torch.cuda.empty_cache()

            if args.run_flaky:
                flaky_rows = load_flaky_rows(args.flaky_file)
                logger.info("running model %s on %d flaky samples", args.modelName, len(flaky_rows))
                s = eval_binary_task(args, generator, tokenizer, flaky_rows, flaky_prompt, gold_key="label",
                                    allowed_labels=["0","1"], task_name="flaky_tests")
                task_summaries.append(s)
            if task_summaries:
                pretty_print_summary(task_summaries)    
            torch.cuda.empty_cache()
            
            if args.run_clone:
                clone_rows = load_clone_rows_txt(args.clone_code_file, args.clone_pairs_file , limit=30000)  
                logger.info("running model %s on %d clone samples", args.modelName, len(clone_rows))
                s = eval_binary_task(
                    args, generator, tokenizer,
                    clone_rows, clone_prompt,
                    gold_key="label",
                    allowed_labels=["0","1"],
                    task_name="clone_detection"
                )
                task_summaries.append(s)
                
                
            if task_summaries:
                pretty_print_summary(task_summaries)      
            torch.cuda.empty_cache()
            
            if args.run_codesearch:
                logger.info("running model %s on code search file: %s", args.modelName, args.codesearch_file)
                s = eval_codesearch_task(args, model, tokenizer, args.codesearch_file, k=args.cs_topk, task_name="code_search")
                task_summaries.append(s)

            if task_summaries:
                pretty_print_summary(task_summaries)

            
            del model, tokenizer, generator
            gc.collect()
            torch.cuda.empty_cache()

