# Development notes

This repo is scaffolded for demonstration and testing. Key points:

- Use `examples/complete_example.py` to run a local demo.
- Configuration lives in `configs/evaluation.yaml`.
- Add real model inference where `Orchestrator._simulate_model` runs.
- Replace placeholder metric wrappers with real `deepeval`/`ragas` packages when available.

Contributions should add tests under `tests/` and CI configuration.
