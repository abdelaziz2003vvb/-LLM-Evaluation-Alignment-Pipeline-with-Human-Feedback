"""RAGAS metrics wrapper (placeholder).

RAGAS is a retrieval-augmented evaluation approach; here we provide a minimal
implementation that computes simple scores to be used in pipelines and tests.
"""
from typing import List, Dict


def compute_ragas_metrics(preds: List[str], ctxs: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute placeholder RAGAS-like metrics.

    `ctxs` represents retrieved context for each prediction. We return a simple
    `context_overlap` and `consistency` score.
    """
    if not preds:
        return {"context_overlap": 0.0, "consistency": 0.0}

    def overlap(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa:
            return 0.0
        return len(sa & sb) / len(sa)

    ctx_overlap = sum(overlap(p, c) for p, c in zip(preds, ctxs)) / len(preds)
    consistency = sum(overlap(p, r) for p, r in zip(preds, refs)) / len(preds)
    return {"context_overlap": float(ctx_overlap), "consistency": float(consistency)}
