"""Tokenizer mapping with ratio-based estimation for duplicate mappings (Eq. 4)."""

from collections import defaultdict

import numpy as np


def map_tokens(ids, mu_b, src, tgt):
    mapped = []

    for v in ids:
        sub = tgt.encode(src.decode([v]), add_special_tokens=False)
        mapped.append(next((u for u in sub if tgt.decode([u]).strip()), sub[0]))

    groups, w = defaultdict(list), np.ones(len(ids))
    for i, u in enumerate(mapped):
        groups[u].append(i)

    for g in groups.values():
        anchor = max(g, key=lambda i: mu_b[i])
        for i in g:
            if i != anchor:
                w[i] = mu_b[i] / max(mu_b[anchor], 1e-30)

    return mapped, w
