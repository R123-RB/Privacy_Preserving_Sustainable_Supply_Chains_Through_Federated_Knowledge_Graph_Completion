"""Shared building blocks for generate_notebook.py (AnonNet) and generate_loadhalal.py.

Both generators embed the *same* source text for the generic parts of the
pipeline (metrics, negative sampling, aggregation simulation, DP accountant,
simulation tests).  Keeping them here means a fix is made once and both
notebooks stay consistent.  `check_notebooks_in_sync.py` regenerates the
notebooks into a temp directory and verifies that the committed .ipynb files
match what these scripts produce.
"""
import json


class Notebook:
    def __init__(self):
        self.cells = []

    @staticmethod
    def _lines(text):
        lines = [line + "\n" for line in text.split("\n")]
        lines[-1] = lines[-1][:-1]
        return lines

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": self._lines(text)})

    def code(self, text):
        self.cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": self._lines(text),
        })

    def write(self, path):
        nb = {
            "cells": self.cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        return len(self.cells)


# ---------------------------------------------------------------------------
# Imports / environment
# ---------------------------------------------------------------------------
IMPORTS_CELL = r'''import os, io, json, math, time, copy, zlib, pickle, random, platform, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
import matplotlib.pyplot as plt

try:                      # `display` exists inside Jupyter; fall back to print elsewhere
    display
except NameError:
    display = print

try:
    import torch_geometric
    import torch_geometric.nn as pyg_nn
    import torch_geometric.utils as pyg_utils
    _HAS_PYG = True
except Exception as _e:
    _HAS_PYG = False
    print(f"torch_geometric not available ({_e}); PyG-only models will be skipped.")

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
print(f"python {platform.python_version()} | torch {torch.__version__} | "
      f"torch_geometric {'%s' % torch_geometric.__version__ if _HAS_PYG else 'n/a'}")'''


# ---------------------------------------------------------------------------
# Metrics (generic over raw score arrays)
# ---------------------------------------------------------------------------
METRICS_CELL = r'''METRIC_NAMES = ['Threshold', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-score',
                'ROC-AUC', 'PR-AUC', 'MRR', 'Hits@10']

def _np1d(a):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64).reshape(-1)

def best_f1_threshold(pos, neg):
    """Decision threshold on RAW scores that maximises F1 -- call this on
    VALIDATION scores only, then reuse the threshold unchanged on test."""
    pos, neg = _np1d(pos), _np1d(neg)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind='mergesort')
    s, y = scores[order], labels[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    boundary = np.r_[np.diff(s) != 0, True]          # only cut between distinct scores
    f1 = np.where(boundary, f1, -1.0)
    k = int(np.argmax(f1))
    return float((s[k] + s[k + 1]) / 2) if k + 1 < len(s) else float(s[k] - 1e-9)

def ranking_metrics(pos, cand):
    """pos: [P] raw scores of the true pairs; cand: [P, K] raw scores of the K
    candidate corruptions of each positive. Ties get the average rank, so a
    degenerate constant scorer cannot look good."""
    pos, cand = _np1d(pos), np.asarray(cand, dtype=np.float64)
    greater = (cand > pos[:, None]).sum(1)
    ties = (cand == pos[:, None]).sum(1)
    rank = 1.0 + greater + 0.5 * ties
    return float(np.mean(1.0 / rank)), float(np.mean(rank <= 10))

def summarize_scores(pos, neg, rank_pos, rank_cand, thr):
    pos, neg = _np1d(pos), _np1d(neg)
    tp = int((pos > thr).sum()); fp = int((neg > thr).sum())
    fn, tn = len(pos) - tp, len(neg) - fp
    prec = tp / max(tp + fp, 1); rec = tp / max(len(pos), 1)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    allsc = np.concatenate([pos, neg])
    mrr, hits10 = ranking_metrics(rank_pos, rank_cand)
    return {
        'Threshold': float(thr),
        'Accuracy': (tp + tn) / (len(pos) + len(neg)),
        'Balanced Accuracy': 0.5 * (rec + tn / max(len(neg), 1)),
        'Precision': prec, 'Recall': rec,
        'F1-score': 2 * prec * rec / max(prec + rec, 1e-12),
        'ROC-AUC': float(roc_auc_score(labels, allsc)),
        'PR-AUC': float(average_precision_score(labels, allsc)),
        'MRR': mrr, 'Hits@10': hits10,
    }

def mean_std_table(per_seed):
    """per_seed: {column_name: [metric_dict, ...]} -> DataFrame with mean / std columns."""
    rows = {'Metric': METRIC_NAMES}
    for col, runs in per_seed.items():
        arr = np.array([[r[m] for m in METRIC_NAMES] for r in runs], dtype=np.float64)
        rows[f'{col} mean'] = arr.mean(0)
        rows[f'{col} std'] = arr.std(0, ddof=1) if len(runs) > 1 else np.zeros(len(METRIC_NAMES))
    return pd.DataFrame(rows)'''


# ---------------------------------------------------------------------------
# Vectorised negative sampling / candidate construction
# ---------------------------------------------------------------------------
SAMPLING_CELL = r'''def make_key_array(pairs, num_nodes, undirected):
    """Sorted int64 keys for a set of (u, v) pairs. undirected=True makes
    (u, v) and (v, u) share one key, so reverse edges cannot slip through."""
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros(0, dtype=np.int64)
    a, b = (np.minimum(p[:, 0], p[:, 1]), np.maximum(p[:, 0], p[:, 1])) if undirected else (p[:, 0], p[:, 1])
    return np.unique(a * num_nodes + b)

def keys_of(u, v, num_nodes, undirected):
    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    if undirected:
        u, v = np.minimum(u, v), np.maximum(u, v)
    return u * num_nodes + v

def in_keys(sorted_keys, q):
    if len(sorted_keys) == 0:
        return np.zeros(len(q), dtype=bool)
    idx = np.searchsorted(sorted_keys, q)
    idx[idx == len(sorted_keys)] = len(sorted_keys) - 1
    return sorted_keys[idx] == q

def sample_negative_pairs(n, rng, src_pool, dst_pool, forbidden_keys, num_nodes, undirected, max_rounds=100):
    """n distinct pairs (u in src_pool, v in dst_pool), u != v, whose key is NOT in
    forbidden_keys. Pools are explicit, so a client can only ever draw its own nodes."""
    src_pool, dst_pool = np.asarray(src_pool, dtype=np.int64), np.asarray(dst_pool, dtype=np.int64)
    got = np.zeros((0, 2), dtype=np.int64)
    for _ in range(max_rounds):
        need = n - len(got)
        if need <= 0:
            break
        m = int(need * 1.5) + 32
        u, v = rng.choice(src_pool, m), rng.choice(dst_pool, m)
        keep = u != v
        u, v = u[keep], v[keep]
        if undirected:
            u, v = np.minimum(u, v), np.maximum(u, v)
        k = u * num_nodes + v
        keep = ~in_keys(forbidden_keys, k)
        cand = np.stack([u[keep], v[keep]], 1)
        got = np.unique(np.concatenate([got, cand]), axis=0)
    if len(got) < n:
        raise RuntimeError(f"could only sample {len(got)}/{n} negatives from the given pools")
    return got[rng.permutation(len(got))[:n]]

def build_ranking_candidates(pos_pairs, k, rng, corrupt_pool, forbidden_keys, num_nodes, undirected, corrupt='tail'):
    """Filtered corruption candidates: for every positive (a, b) keep one endpoint and replace
    the other by k random nodes from `corrupt_pool`; candidates equal to ANY known positive are
    rejected (filtered protocol). corrupt='tail' always replaces b (typed KGC, e.g. which
    manufacturer supplies this food); 'either' flips a coin per positive (untyped graph).
    Returns int64 array [P, k, 2]."""
    pos = np.asarray(pos_pairs, dtype=np.int64).reshape(-1, 2)
    P = len(pos)
    keep_first = np.ones(P, dtype=bool) if corrupt == 'tail' else rng.random(P) < 0.5
    anchor = np.where(keep_first, pos[:, 0], pos[:, 1])
    A = np.repeat(anchor, k)
    B = rng.choice(np.asarray(corrupt_pool, dtype=np.int64), P * k)
    bad = np.ones(P * k, dtype=bool)
    for _ in range(200):
        idx = np.where(bad)[0]
        if len(idx) == 0:
            break
        B[idx] = rng.choice(np.asarray(corrupt_pool, dtype=np.int64), len(idx))
        bad[idx] = (A[idx] == B[idx]) | in_keys(forbidden_keys, keys_of(A[idx], B[idx], num_nodes, undirected))
    if bad.any():
        raise RuntimeError("could not build filtered ranking candidates")
    cand = np.stack([A, B], 1).reshape(P, k, 2)
    if corrupt == 'tail':
        return cand
    # untyped graph: restore (first, second) order irrelevant because scoring is symmetric
    return cand'''


# ---------------------------------------------------------------------------
# Aggregation simulation + DP + tests
# ---------------------------------------------------------------------------
AGGREGATION_CELL = r'''# ======================================================================
# What this cell is -- and is NOT
# ----------------------------------------------------------------------
# * PairwiseMaskSimulation is a SINGLE-PROCESS SIMULATION of additive pairwise
#   masking (the arithmetic idea behind Secure Aggregation). Each client masks
#   its own update using seeds it shares with each other client; the "server"
#   function only ever sums the masked payloads. The masks cancel exactly.
# * It is NOT a deployable Secure Aggregation protocol: there is no real key
#   agreement, no authentication, no dropout recovery and no malicious-party
#   handling, and everything runs in one Python process. It therefore does NOT
#   demonstrate that a real server cannot inspect individual client updates.
# * What is exchanged are model DELTAS (local state - global state), pre-weighted
#   by the aggregation weight.
# * Differential privacy is OFF unless DP_SIGMA > 0. When on, it is client-level
#   DP-FedAvg (clip each client delta to DP_CLIP_NORM, uniform weights, Gaussian
#   noise on the sum) with the Renyi accountant below -- see its assumptions.
# ======================================================================
def flat_norm(delta):
    return float(torch.sqrt(sum((v.double() ** 2).sum() for v in delta.values())))

def clip_delta(delta, clip_norm):
    nrm = flat_norm(delta)
    scale = min(1.0, clip_norm / (nrm + 1e-12))
    return {k: v * scale for k, v in delta.items()}, nrm

class PairwiseMaskSimulation:
    def __init__(self, num_clients, seed):
        rng = np.random.default_rng(seed + 991)
        self.num_clients = num_clients
        # secrets[i][j] is known only to clients i and j (stand-in for a key agreement).
        self._secrets = {i: {} for i in range(num_clients)}
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                s = int(rng.integers(1, 2 ** 61))
                self._secrets[i][j] = s
                self._secrets[j][i] = s

    def client_mask(self, i, payload, round_idx):
        """Runs on client i: uses ONLY client i's own pair secrets."""
        masked = {}
        for key, v in payload.items():
            m = v.double().clone()
            for j, s in self._secrets[i].items():
                seed = (s + 1_000_003 * round_idx + zlib.crc32(key.encode())) % (2 ** 62)
                g = torch.Generator().manual_seed(seed)
                r = torch.randn(v.shape, generator=g, dtype=torch.float64)
                m += r if i < j else -r
            masked[key] = m
        return masked

def server_sum(payloads):
    """Runs on the server: sees only the (masked) payloads."""
    out = {}
    for key in payloads[0]:
        out[key] = sum(p[key] for p in payloads)
    return out

def aggregate_round(client_results, round_idx, masker=None, use_masking=None, dp_sigma=None, clip_norm=None):
    """client_results: list of (delta_dict, n_samples). Returns (aggregate_delta_dict, info)."""
    use_masking = PAIRWISE_MASK_SIM if use_masking is None else use_masking
    dp_sigma = DP_SIGMA if dp_sigma is None else dp_sigma
    clip_norm = DP_CLIP_NORM if clip_norm is None else clip_norm
    K = len(client_results)
    counts = [n for _, n in client_results]
    total = sum(counts)
    if total == 0:
        return None, {}
    use_dp = dp_sigma > 0
    weights = [1.0 / K] * K if use_dp else [n / total for n in counts]
    prepared, pre_clip = [], []
    for (delta, n), w in zip(client_results, weights):
        if use_dp:
            delta, nrm = clip_delta(delta, clip_norm)
            pre_clip.append(nrm)
        prepared.append({k: v.double() * w for k, v in delta.items()})
    payloads = [masker.client_mask(i, p, round_idx) for i, p in enumerate(prepared)] if use_masking else prepared
    agg = server_sum(payloads)
    if use_dp:   # sensitivity of the uniformly-weighted sum to one client is clip_norm / K
        std = dp_sigma * clip_norm / K
        agg = {k: v + torch.randn(v.shape, dtype=torch.float64) * std for k, v in agg.items()}
    # measured size of one client's uplink message (float32 serialisation, in-process)
    uplink = len(pickle.dumps({k: v.float().numpy() for k, v in payloads[0].items()}))
    return {k: v.float() for k, v in agg.items()}, {'uplink_bytes_per_client': uplink,
                                                     'pre_clip_norms': pre_clip}

def dp_epsilon(noise_multiplier, rounds, delta):
    """(epsilon, delta) after `rounds` releases of a Gaussian mechanism with the given noise
    multiplier (std / sensitivity). Renyi DP of the Gaussian mechanism is alpha/(2 sigma^2) per
    release, composed linearly, converted with eps = RDP(alpha) + log(1/delta)/(alpha-1).
    ASSUMPTIONS: privacy unit = one client's entire local dataset (add/remove-one-client);
    every client participates every round, so there is NO subsampling amplification; noise is
    added to the aggregate by the server; model selection/hyper-parameter search is NOT
    accounted for; the validation/test graph used for scoring is outside the privacy accounting."""
    if noise_multiplier <= 0:
        return float('inf')
    alphas = np.concatenate([np.linspace(1.01, 10, 400), np.linspace(10, 4096, 800)])
    eps = rounds * alphas / (2.0 * noise_multiplier ** 2) + math.log(1.0 / delta) / (alphas - 1.0)
    return float(eps.min())

def run_privacy_simulation_tests():
    """Arithmetic checks of the SIMULATION only (not a security analysis)."""
    torch.manual_seed(0)
    shapes = {'a': (5, 3), 'b': (7,), 'c.weight': (4, 4)}
    results = [({k: torch.randn(*s) for k, s in shapes.items()}, n) for n in (10, 20, 0, 5)]
    masker = PairwiseMaskSimulation(len(results), seed=123)
    plain, _ = aggregate_round(results, 1, masker, use_masking=False, dp_sigma=0.0)
    masked, _ = aggregate_round(results, 1, masker, use_masking=True, dp_sigma=0.0)
    err = max(float((plain[k] - masked[k]).abs().max()) for k in plain)
    assert err < 1e-5, f"masked aggregate differs from plain aggregate by {err}"
    total = sum(n for _, n in results)
    expected = {k: sum(d[k] * (n / total) for d, n in results) for k in shapes}
    assert max(float((plain[k] - expected[k]).abs().max()) for k in shapes) < 1e-5
    # an individual masked payload must look nothing like the true contribution
    w0 = {k: v.double() * (results[0][1] / total) for k, v in results[0][0].items()}
    m0 = masker.client_mask(0, w0, 1)
    rel = flat_norm({k: m0[k] - w0[k] for k in w0}) / flat_norm(w0)
    assert rel > 1.0, "masked payload is too close to the true update"
    # clipping
    clipped, _ = clip_delta(results[0][0], 0.5)
    assert flat_norm(clipped) <= 0.5 + 1e-6
    # accountant sanity: more noise -> smaller epsilon; no noise -> infinite epsilon
    assert dp_epsilon(2.0, 20, 1e-5) < dp_epsilon(1.0, 20, 1e-5) < dp_epsilon(0.5, 20, 1e-5)
    assert dp_epsilon(0.0, 20, 1e-5) == float('inf')
    print(f"[simulation tests passed] max |masked - plain| = {err:.2e}; "
          f"masked/true update relative distance = {rel:.1f}")
    print("NOTE: these tests check the masking ARITHMETIC in one process; they are not a security proof.")

run_privacy_simulation_tests()'''
