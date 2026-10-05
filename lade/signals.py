"""Extracting latent safety signals from dark knowledge (Eq. 2-3)."""

import string

import numpy as np

from .model import first_token_probs


def extract_signals(model, tok, harmful, benign, k=500):
    mu_h = sum(first_token_probs(model, tok, q) for q in harmful) / len(harmful)
    mu_b = sum(first_token_probs(model, tok, q) for q in benign) / len(benign)
    special, ids = set(tok.all_special_ids), []

    for v in map(int, np.argsort(np.abs(mu_h - mu_b))[::-1]):
        s = tok.decode([v]).strip()
        if v not in special and s and not all(c in string.punctuation for c in s):
            ids.append(v)
        if len(ids) == k:
            break

    return ids, mu_b[ids]
