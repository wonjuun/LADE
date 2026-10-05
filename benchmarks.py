"""Harmful, benign and jailbreak benchmarks used in the paper (file layout in README)."""

import json
import os

import numpy as np
import pandas as pd

from lade import MODELS

DATA = {
    "advbench": ("advbench.csv", "goal"),
    "hex_phi": ("hex_phi.csv", "prompt"),
    "strongreject": ("strongreject.csv", "forbidden_prompt"),
    "mmlu": ("mmlu.csv", "prompt"),
    "alpaca": ("alpaca.jsonl", "instruction"),
    "gsm8k": ("gsm8k.jsonl", "question"),
    "xstest": ("xstest.csv", "prompt"),
}
ATTACKS = ["autodan", "deepinception", "gcg", "pair", "liar"]
HARMFUL = {"advbench", "hex_phi", "strongreject", *ATTACKS}
OFFLINE = {"hex_phi", "xstest"}


def read(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(l) for l in f] if path.endswith(".jsonl") else json.load(f))


def clean(texts):
    return [q for q in (str(x).strip() for x in texts if not pd.isna(x)) if q]


def to_prompts(name, df):
    s = lambda x: str(x).strip()
    rows = df.itertuples()
    if name == "mmlu":
        return clean(f"{s(r.prompt)}\n\nA. {s(r.A)}\nB. {s(r.B)}\nC. {s(r.C)}\nD. {s(r.D)}" for r in rows)
    if name == "alpaca":
        return clean(f"{s(r.instruction)}\n\n{s(r.input)}" if s(r.input) else s(r.instruction) for r in rows)
    return clean(df[DATA[name][1]])


def load_data(data_dir, seed=42):
    raw = {n: read(os.path.join(data_dir, f)) for n, (f, _) in DATA.items()}
    rng = np.random.RandomState(seed)
    alpaca = sorted(rng.permutation(len(raw["alpaca"]))[:500])
    ref_rows = sorted(rng.permutation(len(raw["hex_phi"]))[:253])
    rows = {"alpaca": alpaca, "mmlu": range(500), "gsm8k": range(500)}
    data = {n: to_prompts(n, df.iloc[list(rows[n])] if n in rows else df) for n, df in raw.items()}
    return data, ref_rows


def load_attacks(data_dir, target):
    key = lambda m: "".join(c for c in str(m).lower() if c.isalnum())
    name_key = key(MODELS.get(target, target).split("/")[-1])
    out = {}

    for name in ATTACKS:
        df = read(os.path.join(data_dir, "attacks", f"{name}.json"))
        if "target_model" in df:
            df = df[df.target_model.map(lambda m: key(m) in name_key)]
        out[name] = clean(df["jailbreak_prompt" if "jailbreak_prompt" in df else "prompt"])

    return out
