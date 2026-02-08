"""Orchestrator: glue between evaluation, feedback, alignment and tracking.

This module provides an `Orchestrator` class that runs an evaluation -> feedback ->
failure mining -> optional fine-tuning loop and logs artifacts/metrics to MLflow.
"""
import yaml
from pathlib import Path
from typing import List
import random

from src.utils.mlflow_tracker import MLflowTracker
from src.evaluation.deepeval_metrics import compute_deepeval_metrics
from src.evaluation.ragas_metrics import compute_ragas_metrics
from src.feedback.feedback_collector import FeedbackCollector
from src.alignment.failure_miner import mine_failures
from src.alignment.fine_tuner import FineTuner


class Orchestrator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mlflow_cfg = cfg.get("mlflow", {})
        self.tracker = MLflowTracker(mlflow_cfg.get("tracking_uri", ""), mlflow_cfg.get("experiment_name", "default"))
        self.model_cfg = cfg.get("model", {})
        self.eval_cfg = cfg.get("evaluation", {})
        self.feedback_cfg = cfg.get("feedback", {})
        self.alignment_cfg = cfg.get("alignment", {})
        self.feedback_collector = FeedbackCollector(self.feedback_cfg.get("feedback_store", "data/feedback.jsonl"))

    def _sample_inputs(self, n: int) -> List[str]:
        # in real systems load dataset; here we simulate prompts
        return [f"Query about topic {i}: what's the fact?" for i in range(n)]

    def _simulate_model(self, prompts: List[str]) -> List[str]:
        # placeholder model responses (replace with real model inference)
        return [p + " Response from model." for p in prompts]

    def evaluate(self):
        n = min(self.eval_cfg.get("max_examples", 50), 50)
        prompts = self._sample_inputs(n)
        preds = self._simulate_model(prompts)
        refs = ["Ground truth about topic." for _ in prompts]
        deepeval_scores = compute_deepeval_metrics(preds, refs)
        ragas_scores = compute_ragas_metrics(preds, ["retrieved context" for _ in preds], refs)
        metrics = {**deepeval_scores, **ragas_scores}
        with self.tracker.start_run("evaluation"):
            self.tracker.log_metrics(metrics)
        return prompts, preds, refs, metrics

    def collect_feedback_interactive(self):
        # launch a CLI collector for demo; in production use Gradio or web UI
        self.feedback_collector.collect_cli()

    def run_full_cycle(self, do_finetune: bool = True):
        prompts, preds, refs, metrics = self.evaluate()
        print("Evaluation metrics:", metrics)

        # For demo, create a few simulated feedback entries
        for i in range(3):
            idx = random.randrange(len(prompts))
            self.feedback_collector.add(prompts[idx], preds[idx], correction="Corrected answer.", rating=2, notes="demo")

        failures_path = self.alignment_cfg.get("failures_output", "data/failures.jsonl")
        n_failures = mine_failures(self.feedback_cfg.get("feedback_store"), failures_path, self.feedback_cfg.get("min_rating_for_retrain", 3))
        print(f"Mined {n_failures} failures to {failures_path}")

        if do_finetune and n_failures > 0:
            tuner = FineTuner(self.model_cfg.get("hf_model_id", "gpt-neo-125M"))
            try:
                out = tuner.fine_tune(failures_path)
                if out:
                    with self.tracker.start_run("finetune"):
                        self.tracker.log_params({"finetuned_model_path": out})
            except Exception as e:
                print("Fine-tuning skipped/failed:", e)

        return metrics
