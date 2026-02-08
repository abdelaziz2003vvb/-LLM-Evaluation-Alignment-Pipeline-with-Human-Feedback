"""Wrappers for DeepEval metrics.

This module attempts to import a `deepeval` package; if unavailable, it provides
lightweight fallback computations for demonstration and testing.
"""
from typing import List, Dict


def compute_deepeval_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute simple placeholder DeepEval-like metrics.

    Returns a dict with `factuality`, `relevance`, and `robustness` scores in [0,1].
    """
    if len(preds) == 0:
        return {"factuality": 0.0, "relevance": 0.0, "robustness": 0.0}

    # simple token overlap heuristics as a lightweight proxy
    def overlap(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa:
            return 0.0
        return len(sa & sb) / len(sa)

    factual = sum(overlap(p, r) for p, r in zip(preds, refs)) / len(preds)
    relevance = factual  # proxy
    robustness = max(0.0, 1.0 - 0.1 * sum(len(p) < 10 for p in preds) / len(preds))

    return {"factuality": float(factual), "relevance": float(relevance), "robustness": float(robustness)}
