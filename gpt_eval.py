#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT (OpenAI Batch API) zero-shot / few-shot evaluation.

gpt-5.6-luna is API-only (no local weights), so it can't reuse zero_shot.py's
/ few_shot.py's local `pipeline(...)` generation loop. This script reuses their
prompt builders and (for few-shot) BM25 demonstration retrieval, but sends
requests through the OpenAI Batch API instead of synchronous local generation.

code_search has no embeddings to retrieve with here (chat-only API, no hidden
states) — it's evaluated as pure listwise LLM re-ranking: every query is
scored against the full sampled candidate pool in chunks, no first-stage
embedding retrieval.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

import zero_shot as zs
import few_shot as fs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("gpt_eval")

MAX_REQUESTS_PER_BATCH = 40000  # headroom under the Batch API's ~50K/file cap
MAX_BATCH_FILE_BYTES = 80_000_000  # headroom under the Batch API's ~100-200MB/file cap


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running, e.g.:\n"
            "  set -a; source /home/aakli/v2_stage/.env; set +a"
        )
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


# ─────────────────────────────── batch helpers ───────────────────────────────

def _chat_request(custom_id: str, model: str, prompt: str, max_completion_tokens: int) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
            # gpt-5.6-luna only supports the default temperature (1) — no override.
        },
    }


def _write_batch_file(requests: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")


def _chunk_requests(requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split into batch files respecting both a request-count cap and a
    byte-size cap (line length varies a lot: 20-candidate rerank prompts vs.
    short binary-classification prompts), whichever is hit first."""
    chunks, cur, cur_bytes = [], [], 0
    for r in requests:
        line_bytes = len(json.dumps(r).encode("utf-8")) + 1
        if cur and (len(cur) >= MAX_REQUESTS_PER_BATCH or cur_bytes + line_bytes > MAX_BATCH_FILE_BYTES):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += line_bytes
    if cur:
        chunks.append(cur)
    return chunks


def submit_and_wait(client: OpenAI, requests: List[Dict[str, Any]], out_dir: str,
                     tag: str, poll_interval: int, dry_run: bool = False) -> Dict[str, str]:
    """Submit `requests` (chunked under the per-batch request/size caps), block
    until every chunk finishes, and return {custom_id: assistant_text}."""
    os.makedirs(out_dir, exist_ok=True)
    chunks = _chunk_requests(requests)

    total_chars = sum(len(json.dumps(r)) for r in requests)
    logger.info("[%s] %d requests in %d batch file(s), ~%.1f MB",
                tag, len(requests), len(chunks), total_chars / 1e6)

    if dry_run:
        for ci, chunk in enumerate(chunks):
            sub_tag = tag if len(chunks) == 1 else f"{tag}_part{ci}"
            _write_batch_file(chunk, os.path.join(out_dir, f"{sub_tag}_requests.jsonl"))
        logger.info("[%s] DRY RUN — batch file(s) written, nothing submitted to the API.", tag)
        return {}

    batches = []
    for ci, chunk in enumerate(chunks):
        sub_tag = tag if len(chunks) == 1 else f"{tag}_part{ci}"
        req_path = os.path.join(out_dir, f"{sub_tag}_requests.jsonl")
        _write_batch_file(chunk, req_path)
        up = client.files.create(file=open(req_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": sub_tag},
        )
        logger.info("[%s] submitted batch %s (%d requests)", sub_tag, batch.id, len(chunk))
        batches.append((sub_tag, batch))

    results: Dict[str, str] = {}
    pending = list(batches)
    while pending:
        time.sleep(poll_interval)
        still_pending = []
        for sub_tag, batch in pending:
            batch = client.batches.retrieve(batch.id)
            if batch.status in ("validating", "in_progress", "finalizing"):
                still_pending.append((sub_tag, batch))
                counts = batch.request_counts
                logger.info("[%s] batch %s status=%s  completed=%s/%s  failed=%s",
                            sub_tag, batch.id, batch.status,
                            getattr(counts, "completed", "?"), getattr(counts, "total", "?"),
                            getattr(counts, "failed", "?"))
            elif batch.status == "completed":
                if batch.output_file_id:
                    content = client.files.content(batch.output_file_id).text
                    for line in content.strip().splitlines():
                        obj = json.loads(line)
                        cid = obj["custom_id"]
                        try:
                            results[cid] = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
                        except (KeyError, IndexError, TypeError):
                            results[cid] = ""
                logger.info("[%s] batch %s completed", sub_tag, batch.id)
            else:
                raise RuntimeError(
                    f"[{sub_tag}] batch {batch.id} ended with status={batch.status!r}, "
                    f"errors_file_id={batch.error_file_id}"
                )
        pending = still_pending

    out_path = os.path.join(out_dir, f"{tag}_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for cid, text in results.items():
            f.write(json.dumps({"custom_id": cid, "text": text}) + "\n")
    logger.info("[%s] all batches complete: %d results -> %s", tag, len(results), out_path)
    return results


# ─────────────────────────────── binary task eval (vuln / flaky / clone) ───────────────────────────────

def _extract_label(text: str) -> str:
    t = (text or "").lower()
    if "not" in t or "no" in t or "safe" in t or "clean" in t:
        return "0"
    return "1"


def run_binary_task(client: OpenAI, rows: List[Dict[str, Any]], prompt_fn, task_name: str,
                     model: str, out_dir: str, poll_interval: int,
                     max_completion_tokens: int = 20, dry_run: bool = False) -> Dict[str, Any]:
    requests = [
        _chat_request(f"{task_name}_{i}", model, prompt_fn(row), max_completion_tokens)
        for i, row in enumerate(rows)
    ]
    results = submit_and_wait(client, requests, out_dir, task_name, poll_interval, dry_run=dry_run)
    if dry_run:
        return {"Model": model, "Task": task_name, "Samples": len(rows), "DryRun": True}

    gold, pred = [], []
    for i, row in enumerate(rows):
        text = results.get(f"{task_name}_{i}", "")
        pred.append(_extract_label(text))
        gold.append(str(int(row["label"])))

    labels_sorted = ["0", "1"]
    acc = accuracy_score(gold, pred)
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, labels=labels_sorted, average="macro", zero_division=0)
    mcc = matthews_corrcoef(gold, pred)

    return {
        "Model": model, "Task": task_name, "Samples": len(gold),
        "Accuracy": round(acc, 4), "Macro_P": round(p, 4), "Macro_R": round(r, 4),
        "Macro_F1": round(f1, 4), "MCC": round(mcc, 4),
    }


# ─────────────────────── code search: listwise LLM re-rank, no embeddings ───────────────────────

_SCORE_LINE = re.compile(r"^\s*(\d+)\s*[\t ,:|-]\s*(\d+(?:\.\d+)?)\s*$")


def _build_rerank_prompt(query: str, ids: List[int], codes: List[str], max_code_chars: int = 800) -> str:
    head = (
        "Act as a code-search re-ranker. For the QUERY and CANDIDATES, "
        "score each candidate 0-100 (100 = exact implementation, 0 = unrelated). "
        "Output one line per candidate: <ID>\\t<score>. No extra text.\n\n"
        f"QUERY:\n---\n{query}\n---\n\nCANDIDATES:\n"
    )
    body = [f"ID={i}\nCODE:\n---\n{code[:max_code_chars]}\n---\n" for i, code in zip(ids, codes)]
    return head + "\n".join(body)


def load_codesearch_rows(path: str, n_samples: int, seed: int) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            js = json.loads(line)
            if isinstance(js.get("function_tokens"), list): code = " ".join(js["function_tokens"])
            elif isinstance(js.get("code_tokens"), list): code = " ".join(js["code_tokens"])
            else: code = " ".join(str(js.get("function", js.get("code", ""))).split())
            if isinstance(js.get("docstring_tokens"), list): nl = " ".join(js["docstring_tokens"])
            else: nl = " ".join(str(js.get("docstring", js.get("doc", ""))).split())
            rows.append({"code": code, "nl": nl})
    if n_samples < len(rows):
        rows = random.Random(seed).sample(rows, n_samples)
    return rows


def run_codesearch_task(client: OpenAI, rows: List[Dict[str, str]], model: str, out_dir: str,
                         poll_interval: int, cands_per_prompt: int = 20,
                         candidate_pool_cap: int = 100, seed: int = 42,
                         max_completion_tokens: int = 1024, dry_run: bool = False) -> Dict[str, Any]:
    """Listwise LLM re-rank, no embedding pre-filter. To keep request volume in
    line with the other tasks, each query is scored against a per-query capped
    candidate pool (its own gold match + random distractors from the sampled
    set) rather than the full pool — full-pool scoring is O(n) requests/query
    and blows up fast (e.g. n=1000 -> 50 chunks/query -> 50K requests total).
    """
    n = len(rows)
    codes = [r["code"] for r in rows]
    rng = random.Random(seed)

    query_candidates: Dict[int, List[int]] = {}
    requests = []
    for qi, row in enumerate(rows):
        distractors = [i for i in range(n) if i != qi]
        cap = min(candidate_pool_cap - 1, len(distractors))
        cand_ids = rng.sample(distractors, cap) + [qi]
        rng.shuffle(cand_ids)  # gold match shouldn't sit at a fixed position
        query_candidates[qi] = cand_ids

        for c0 in range(0, len(cand_ids), cands_per_prompt):
            chunk_ids = cand_ids[c0:c0 + cands_per_prompt]
            prompt = _build_rerank_prompt(row["nl"], chunk_ids, [codes[i] for i in chunk_ids])
            requests.append(_chat_request(f"cs_{qi}_{c0}", model, prompt, max_completion_tokens))

    logger.info("code_search: %d queries x <=%d candidates/query (capped, no embedding pre-filter) "
                "-> %d requests", n, candidate_pool_cap, len(requests))

    results = submit_and_wait(client, requests, out_dir, "code_search", poll_interval, dry_run=dry_run)
    if dry_run:
        return {"Model": model, "Task": "code_search", "Samples": n, "Requests": len(requests), "DryRun": True}

    ranks = []
    for qi in range(n):
        cand_ids = query_candidates[qi]
        scores: Dict[int, float] = {}
        for c0 in range(0, len(cand_ids), cands_per_prompt):
            text = results.get(f"cs_{qi}_{c0}", "")
            for line in text.strip().splitlines():
                m = _SCORE_LINE.match(line)
                if m:
                    idx = int(m.group(1))
                    if idx in cand_ids:  # guard against a hallucinated/out-of-chunk id
                        scores[idx] = max(0.0, min(100.0, float(m.group(2))))
        order = sorted(cand_ids, key=lambda i: scores.get(i, -1.0), reverse=True)
        rank = order.index(qi) + 1  # gold code for query qi is codes[qi] (paired by construction)
        ranks.append(1.0 / rank)

    mrr = float(np.mean(ranks)) if ranks else 0.0
    return {"Model": model, "Task": "code_search", "Samples": n, "Queries": n,
            "CandidatePool": candidate_pool_cap, "MRR@K": round(mrr, 4)}


def pretty_print_summary(task_summaries: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(task_summaries)
    col_order = [c for c in ["Task", "Model", "Samples", "Accuracy", "Macro_F1", "MCC", "Queries", "MRR@K"]
                if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]
    print("\n" + "=" * 68)
    print("GPT EVALUATION SUMMARY".center(68))
    print("=" * 68)
    print(df.to_string(index=False))
    print("=" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gpt-5.6-luna (OpenAI Batch API) zero-shot or few-shot")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--mode", choices=["zero", "few"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1000, help="Samples evaluated per benchmark task.")
    parser.add_argument("--poll_interval", type=int, default=30, help="Seconds between batch status polls.")
    parser.add_argument("--out_dir", default="./gpt_batch_runs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build batch request files and print sizing, but never call the API.")

    parser.add_argument("--run_vuln", action="store_true")
    parser.add_argument("--vuln_file", default="./datasets/dataset_vulnerabilty/test.jsonl")
    parser.add_argument("--vuln_pool_file", default="./datasets/dataset_vulnerabilty/train.jsonl")

    parser.add_argument("--run_flaky", action="store_true")
    parser.add_argument("--flaky_file", default="./datasets/dataset_flakytest/valid.json")
    parser.add_argument("--flaky_pool_file", default="./datasets/dataset_flakytest/train.json")

    parser.add_argument("--run_clone", action="store_true")
    parser.add_argument("--clone_code_file", default="./datasets/dataset_clone/data.jsonl")
    parser.add_argument("--clone_pairs_file", default="./datasets/dataset_clone/test.txt")
    parser.add_argument("--clone_pool_pairs_file", default="./datasets/dataset_clone/train.txt")

    parser.add_argument("--run_codesearch", action="store_true")
    parser.add_argument("--codesearch_file", default="./datasets/code_search/test.jsonl")
    parser.add_argument("--cs_cands_per_prompt", type=int, default=20)
    parser.add_argument("--cs_candidate_pool_cap", type=int, default=100,
                        help="Max candidates scored per query (gold match + random distractors), "
                             "not the full sampled pool -- bounds request volume.")

    parser.add_argument("--n_shots", type=int, default=5, help="Few-shot demonstrations per query (--mode few only).")
    parser.add_argument("--shot_pool_cap", type=int, default=5000)

    args = parser.parse_args()
    random.seed(args.seed)

    client = None if args.dry_run else get_client()
    out_dir = os.path.join(args.out_dir, args.mode)
    task_summaries = []

    if args.run_vuln:
        if args.mode == "zero":
            all_rows = zs.load_vuln_rows(args.vuln_file)
            rows = random.sample(all_rows, args.n_samples) if args.n_samples < len(all_rows) else all_rows
            prompt_fn = zs.vuln_prompt
        else:
            rows = fs.load_vuln_rows(args.vuln_file, args.vuln_pool_file, n_shots=args.n_shots,
                                     pool_cap=args.shot_pool_cap, seed=args.seed, limit=args.n_samples)
            prompt_fn = fs.vuln_prompt
        logger.info("vulnerability_detection: %d samples (mode=%s)", len(rows), args.mode)
        s = run_binary_task(client, rows, prompt_fn, "vulnerability_detection", args.model,
                            out_dir, args.poll_interval, dry_run=args.dry_run)
        task_summaries.append(s)
        pretty_print_summary(task_summaries)

    if args.run_flaky:
        if args.mode == "zero":
            all_rows = zs.load_flaky_rows(args.flaky_file)
            rows = random.sample(all_rows, args.n_samples) if args.n_samples < len(all_rows) else all_rows
            prompt_fn = zs.flaky_prompt
        else:
            rows = fs.load_flaky_rows(args.flaky_file, args.flaky_pool_file, n_shots=args.n_shots,
                                      pool_cap=args.shot_pool_cap, seed=args.seed, limit=args.n_samples)
            prompt_fn = fs.flaky_prompt
        logger.info("flaky_tests: %d samples (mode=%s)", len(rows), args.mode)
        s = run_binary_task(client, rows, prompt_fn, "flaky_tests", args.model,
                            out_dir, args.poll_interval, dry_run=args.dry_run)
        task_summaries.append(s)
        pretty_print_summary(task_summaries)

    if args.run_clone:
        if args.mode == "zero":
            all_rows = zs.load_clone_rows_txt(args.clone_code_file, args.clone_pairs_file)
            rows = random.sample(all_rows, args.n_samples) if args.n_samples < len(all_rows) else all_rows
            prompt_fn = zs.clone_prompt
        else:
            rows = fs.load_clone_rows_txt(args.clone_code_file, args.clone_pairs_file, args.clone_pool_pairs_file,
                                          limit=args.n_samples, n_shots=args.n_shots,
                                          pool_cap=args.shot_pool_cap, seed=args.seed)
            prompt_fn = fs.clone_prompt
        logger.info("clone_detection: %d samples (mode=%s)", len(rows), args.mode)
        s = run_binary_task(client, rows, prompt_fn, "clone_detection", args.model,
                            out_dir, args.poll_interval, dry_run=args.dry_run)
        task_summaries.append(s)
        pretty_print_summary(task_summaries)

    if args.run_codesearch:
        rows = load_codesearch_rows(args.codesearch_file, args.n_samples, args.seed)
        logger.info("code_search: %d samples (listwise re-rank, no embeddings)", len(rows))
        s = run_codesearch_task(client, rows, args.model, out_dir, args.poll_interval,
                                cands_per_prompt=args.cs_cands_per_prompt,
                                candidate_pool_cap=args.cs_candidate_pool_cap,
                                seed=args.seed, dry_run=args.dry_run)
        task_summaries.append(s)
        pretty_print_summary(task_summaries)
