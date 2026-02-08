"""Simple human feedback collector utilities.

This module supports programmatic (CLI) collection and an optional Gradio UI.
Feedback is stored as JSONL lines with fields: `id`, `input`, `prediction`, `correction`, `rating`, `notes`.
"""
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any


class FeedbackCollector:
    def __init__(self, store_path: str):
        self.store = Path(store_path)
        self.store.parent.mkdir(parents=True, exist_ok=True)
        if not self.store.exists():
            self.store.write_text("")

    def add(self, input_text: str, prediction: str, correction: Optional[str] = None, rating: Optional[int] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()),
            "input": input_text,
            "prediction": prediction,
            "correction": correction,
            "rating": rating,
            "notes": notes,
        }
        with self.store.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def load_all(self):
        with self.store.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    # lightweight CLI helper
    def collect_cli(self):
        print("Collecting feedback (empty input to stop)")
        while True:
            inp = input("Input prompt: ")
            if not inp:
                break
            pred = input("Model prediction: ")
            corr = input("Correction (optional): ")
            rating_s = input("Rating 1-5 (optional): ")
            rating = int(rating_s) if rating_s.strip() else None
            notes = input("Notes (optional): ")
            item = self.add(inp, pred, corr or None, rating, notes or None)
            print("Saved feedback:", item["id"])
