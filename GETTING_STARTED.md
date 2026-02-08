# Getting Started

1. Create a Python environment (recommended: venv or conda)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Start an MLflow server if you want tracking (optional):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

3. Run the example:

```bash
python examples/complete_example.py
```

4. Inspect `data/feedback.jsonl` and run the dashboard with Gradio (if installed).
