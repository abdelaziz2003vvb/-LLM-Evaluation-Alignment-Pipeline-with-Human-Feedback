# LLM Evaluation & Alignment Pipeline with Human Feedback

An **end-to-end autonomous pipeline** for evaluating, aligning, and continuously improving large language models through human feedback loops. Built with Python, PyTorch, LangChain, DeepEval, RAGAS, Gradio, and MLflow.

## Overview

This project implements a production-ready framework for:

- **Automated Evaluation**: Compute factuality, relevance, and robustness scores using DeepEval and RAGAS metrics
- **Human-in-the-Loop Feedback**: Collect corrections, ratings, and preference signals via CLI or Gradio UI
- **Failure Mining**: Automatically extract low-quality or corrected examples for retraining
- **Continuous Fine-Tuning**: Align models using LoRA (Low-Rank Adaptation) for memory-efficient training
- **Experiment Tracking**: Full MLflow integration for metrics, artifacts, and model versioning
- **Interactive Dashboard**: Gradio-based UI for inspecting feedback and evaluation results

---

## Key Features

✅ **Scalable Evaluation Framework**
- Pluggable metric modules (DeepEval, RAGAS with fallback implementations)
- Batch evaluation on arbitrary test sets
- MLflow tracking for all runs and metrics

✅ **Feedback Collection**
- Lightweight JSONL-based storage
- CLI collector for rapid iteration
- Gradio dashboard for web-based feedback review

✅ **Alignment Loop**
- Automatic failure case mining from feedback
- LoRA-based fine-tuning (efficient on limited hardware)
- Integration with Hugging Face transformers and datasets

✅ **MLOps Ready**
- Experiment tracking and comparison
- Model artifact management
- Metric logging and visualization

---

## Architecture

```
llm-eval-pipeline/
├─ src/
│  ├─ evaluation/           # Metric computation (DeepEval, RAGAS)
│  │  ├─ deepeval_metrics.py
│  │  └─ ragas_metrics.py
│  ├─ feedback/             # Human feedback collection
│  │  └─ feedback_collector.py
│  ├─ alignment/            # Fine-tuning & failure mining
│  │  ├─ failure_miner.py
│  │  └─ fine_tuner.py (LoRA-based)
│  ├─ dashboards/           # UI components
│  │  └─ feedback_dashboard.py (Gradio)
│  ├─ pipeline/             # Orchestration
│  │  └─ orchestrator.py
│  └─ utils/                # Utilities
│     └─ mlflow_tracker.py
├─ configs/                 # Configuration files
│  └─ evaluation.yaml       # Pipeline config
├─ examples/
│  └─ complete_example.py   # End-to-end demo
├─ data/                    # Generated at runtime
│  ├─ feedback.jsonl        # Collected feedback
│  └─ failures.jsonl        # Mined failures
└─ outputs/                 # Model outputs
   └─ finetuned/           # LoRA fine-tuned models
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11+ (optional, for GPU acceleration)

### Setup

1. **Clone and navigate**:
```bash
cd llm-eval-pipeline
```

2. **Create virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **(Optional) Install LoRA support**:
```bash
pip install peft
```

---

## Quick Start

### 1. Start MLflow Server (Optional but Recommended)
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```
Then access the UI at `http://localhost:5000`

### 2. Run Complete Pipeline
```bash
python examples/complete_example.py
```

This will:
- ✅ Evaluate sample prompts with DeepEval & RAGAS metrics
- ✅ Log metrics to MLflow
- ✅ Collect simulated feedback
- ✅ Mine failure cases
- ✅ Fine-tune model with LoRA
- ✅ Track all artifacts and models

### 3. Inspect Feedback (Optional)
```bash
cat data/feedback.jsonl
```

### 4. Launch Gradio Dashboard (Optional)
```python
from src.dashboards.feedback_dashboard import run_dashboard
run_dashboard(port=7860)
```

---

## Configuration

Edit `configs/evaluation.yaml` to customize:

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "llm-eval-pipeline"

model:
  name: "gpt-neo-125m"
  hf_model_id: "EleutherAI/gpt-neo-125M"

evaluation:
  batch_size: 8
  max_examples: 200
  metrics: [deepeval, ragas]

feedback:
  feedback_store: "data/feedback.jsonl"
  min_rating_for_retrain: 3

alignment:
  failures_output: "data/failures.jsonl"
```

---

## Component Details

### Evaluation (`src/evaluation/`)
- **DeepEval**: Factuality, relevance, and robustness scoring
- **RAGAS**: Retrieval-augmented evaluation (context overlap, consistency)
- Fallback implementations for testing without external APIs

### Feedback Collection (`src/feedback/`)
- **FeedbackCollector**: JSONL-based storage for ratings, corrections, and notes
- **CLI interface**: `collect_cli()` for interactive feedback entry
- **Programmatic API**: Add feedback directly from code

### Alignment (`src/alignment/`)
- **FailureMiner**: Extract low-rated or corrected examples from feedback
- **FineTuner**: LoRA-based fine-tuning using Hugging Face Trainer
  - Targets GPT-Neo's `c_attn` and `c_proj` modules
  - Memory-efficient training on consumer hardware
  - Auto-saves models to `outputs/finetuned/`

### Orchestrator (`src/pipeline/orchestrator.py`)
- Coordinates evaluation → feedback → mining → fine-tuning
- Integrates MLflow logging for reproducibility
- Extensible design for custom workflows

---

## Usage Examples

### Programmatic Fine-Tuning
```python
from src.alignment.fine_tuner import FineTuner

tuner = FineTuner(hf_model_id="EleutherAI/gpt-neo-125M")
output_path = tuner.fine_tune("data/failures.jsonl", epochs=2, use_lora=True)
print(f"Fine-tuned model saved to: {output_path}")
```

### Manual Feedback Collection
```python
from src.feedback.feedback_collector import FeedbackCollector

collector = FeedbackCollector("data/feedback.jsonl")
collector.add(
    input_text="What is AI?",
    prediction="AI is artificial intelligence.",
    correction="AI is artificial intelligence, a branch of computer science.",
    rating=2,
    notes="Missing detail about CS branch"
)
```

### Mining Failures
```python
from src.alignment.failure_miner import mine_failures

count = mine_failures("data/feedback.jsonl", "data/failures.jsonl", min_rating_for_retrain=3)
print(f"Mined {count} failure cases")
```

---

## MLflow Integration

All runs are automatically logged to MLflow:

```
Metrics logged:
  - factuality, relevance, robustness (DeepEval)
  - context_overlap, consistency (RAGAS)
  - training loss, validation metrics

Artifacts:
  - Fine-tuned models
  - Failure datasets
  - Config snapshots
```

Access the tracking UI at `http://localhost:5000`

---

## LoRA Fine-Tuning

This project uses **LoRA (Low-Rank Adaptation)** for efficient model alignment:

- **Memory efficient**: Train on GPUs with <8GB VRAM
- **Fast**: 2-4x faster than full fine-tuning
- **Modular**: Adapter weights can be merged or swapped

Target modules for GPT-Neo:
- `c_attn`: Multi-head attention
- `c_proj`: Output projection

Configure in `src/alignment/fine_tuner.py`:
```python
lora_config = LoraConfig(
    r=8,                           # Rank
    lora_alpha=32,                 # Scaling
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
)
```

---

## Data Formats

### Feedback (JSONL)
```json
{"id": "uuid", "input": "prompt", "prediction": "model output", "correction": "corrected text", "rating": 2, "notes": "feedback notes"}
```

### Failures (JSONL)
```json
{"id": "uuid", "input": "prompt", "prediction": "model output", "correction": "corrected text", "rating": 1, "notes": "notes"}
```

---

## Troubleshooting

**ModuleNotFoundError: No module named 'src'**
- Ensure you're running from the project root directory
- Virtual environment must be activated

**MLflow connection error**
- Start MLflow server: `mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000`

**LoRA target modules not found**
- Verify model architecture matches expected module names
- GPT-Neo uses `c_attn` and `c_proj`; adjust for other models

**Out of memory during fine-tuning**
- Reduce `batch_size` in config
- Use LoRA (automatic with `use_lora=True`)
- Reduce `max_length` in tokenizer

---

## Project Structure & Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Contribution guidelines
- Adding custom metrics
- Extending feedback collection
- Testing strategies

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup.

---

## Requirements

Core dependencies:
- `pyyaml` - Configuration management
- `mlflow` - Experiment tracking
- `transformers` - HF models
- `datasets` - Data loading
- `torch` - Deep learning
- `peft` - LoRA implementation
- `gradio` - Interactive UI
- `deepeval`, `ragas` - Evaluation metrics (optional)

See `requirements.txt` for full list.

---

## License

MIT License - see LICENSE file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_eval_pipeline_2026,
  title = {LLM Evaluation & Alignment Pipeline with Human Feedback},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/llm-eval-pipeline}
}
```

---

## Support & Contributing

- **Issues**: Report bugs via GitHub Issues
- **PRs**: Contributions welcome! Please follow development guidelines
- **Questions**: Check GETTING_STARTED.md and DEVELOPMENT.md first

---

**Last Updated**: February 2026
