"""Offline stage of LADE on a reference LLM and evaluation on target LLMs.

    python run.py --reference llama3 --targets llama3 llama2 qwen2 qwen3 gemma mistral
"""

import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from benchmarks import HARMFUL, OFFLINE, load_attacks, load_data
from lade import (
    MODELS,
    KNNDetector,
    extract_signals,
    featurize,
    first_token_probs,
    load_model,
    load_tokenizer,
    map_tokens,
)


def cached_signals(path, reference, tok, harmful, benign, k):
    if not os.path.exists(path):
        model = load_model(reference)
        ids, mu_b = extract_signals(model, tok, harmful, benign, k)
        with open(path, "w") as f:
            json.dump({"tokens": ids, "decoded": [tok.decode([v]) for v in ids], "mu_b": mu_b.tolist()}, f)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(path) as f:
        return json.load(f)


def cached_probs(path, target, tok, tokens, sets):
    P = dict(np.load(path)) if os.path.exists(path) else {}
    missing = [n for n in sets if n not in P]

    if missing:
        print(f"[forward] {target}: {missing}", flush=True)
        model = load_model(target)
        for n in missing:
            P[n] = np.stack([first_token_probs(model, tok, q)[tokens] for q in sets[n]])
        np.savez(path, **P)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="llama3")
    ap.add_argument("--targets", nargs="+", default=list(MODELS))
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--rho", type=float, default=0.90)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--attacks", action="store_true", help="Also evaluate data/attacks/*.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = os.path.join(a.out, f"{a.reference}_k{a.k}")
    os.makedirs(out, exist_ok=True)
    data, ref_rows = load_data(a.data_dir, a.seed)
    ref_tok = load_tokenizer(a.reference)

    harmful = [data["hex_phi"][i] for i in ref_rows]
    sig_path = os.path.join(out, "signals.json")
    sig = cached_signals(sig_path, a.reference, ref_tok, harmful, data["xstest"], a.k)
    print(f"[signals] {a.reference}: {sig['decoded'][:10]}", flush=True)

    rows = []
    for t in a.targets:
        tok = load_tokenizer(t)
        if t == a.reference:
            tokens, w = sig["tokens"], np.ones(a.k)
        else:
            tokens, w = map_tokens(sig["tokens"], np.array(sig["mu_b"]), ref_tok, tok)

        sets = {**data, **(load_attacks(a.data_dir, t) if a.attacks else {})}
        P = cached_probs(os.path.join(out, f"{t}.npz"), t, tok, tokens, sets)
        det = KNNDetector(featurize(P["hex_phi"][ref_rows], w), a.rho, a.K, a.n_boot, a.seed)

        for n in sets:
            blocked = det(featurize(P[n], w))
            acc = float((blocked == (n in HARMFUL)).mean())
            rows.append(dict(target=t, dataset=n, accuracy=acc, n=len(blocked)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"results_rho{a.rho}_K{a.K}.csv"), index=False)
    table = df.pivot(index="dataset", columns="target", values="accuracy").reindex(df.dataset.unique())
    table.loc["held-out avg"] = table.loc[[n for n in table.index if n not in OFFLINE]].mean()
    print(table[a.targets].round(4).to_string())


if __name__ == "__main__":
    main()
