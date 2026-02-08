"""Failure mining utilities: extract examples from feedback for retraining.
"""
import json
from pathlib import Path
from typing import Iterable, Dict


def mine_failures(feedback_path: str, output_path: str, min_rating_for_retrain: int = 3) -> int:
    """Scan a JSONL feedback file and write failure cases to output_path.

    Returns number of failures written.
    """
    inp = Path(feedback_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                item = json.loads(line)
            except Exception:
                continue
            rating = item.get("rating")
            correction = item.get("correction")
            # treat low ratings or explicit corrections as failures
            if (rating is not None and rating < min_rating_for_retrain) or correction:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count
