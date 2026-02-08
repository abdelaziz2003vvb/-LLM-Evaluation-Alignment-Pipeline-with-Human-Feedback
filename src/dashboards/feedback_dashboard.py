"""A minimal Gradio dashboard to inspect collected feedback.

This dashboard reads the JSONL feedback store and shows inputs, predictions,
corrections and ratings. It is intended for local inspection and demo purposes.
"""
import json
from pathlib import Path
from typing import List

try:
    import gradio as gr
except Exception:
    gr = None


def _load_feedback(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def run_dashboard(feedback_path: str = "data/feedback.jsonl", port: int = 7860):
    if gr is None:
        raise RuntimeError("Gradio is not installed")
    data = _load_feedback(feedback_path)

    def get_preview(idx: int):
        if idx < 0 or idx >= len(data):
            return {}, "Index out of range"
        return data[idx], f"Showing {idx+1}/{len(data)}"

    with gr.Blocks() as demo:
        gr.Markdown("# Feedback Dashboard")
        idx = gr.Slider(minimum=0, maximum=max(0, len(data)-1), step=1, label="Index")
        preview = gr.JSON()
        status = gr.Textbox()
        idx.change(lambda i: get_preview(int(i)), inputs=idx, outputs=[preview, status])

    demo.launch(server_port=port)
