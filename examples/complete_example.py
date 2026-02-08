"""Run a simple end-to-end demonstration of the pipeline.

This script loads the YAML config, instantiates the orchestrator, runs evaluation,
collects simulated feedback, mines failures and (optionally) fine-tunes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.pipeline.orchestrator import Orchestrator


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "evaluation.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    orch = Orchestrator(cfg)
    orch.run_full_cycle()


if __name__ == "__main__":
    main()
