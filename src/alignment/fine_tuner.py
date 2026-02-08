"""Lightweight fine-tuner using Hugging Face transformers Trainer with LoRA support.

This trainer is minimal and intended for small-scale local fine-tuning. It expects
failure cases as JSONL with fields `input` and `correction` or `prediction`.
Uses LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.
"""
from pathlib import Path
from typing import Optional
import json

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from datasets import Dataset
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    Trainer = None
    TrainingArguments = None
    Dataset = None

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class FineTuner:
    def __init__(self, hf_model_id: str, output_dir: str = "outputs/finetuned"):
        self.hf_model_id = hf_model_id
        self.output_dir = output_dir

    def _load_dataset(self, failures_path: str):
        records = []
        p = Path(failures_path)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                input_text = j.get("input")
                correction = j.get("correction") or j.get("prediction")
                if input_text and correction:
                    records.append({"input": input_text, "target": correction})
        if not records:
            return None
        return Dataset.from_list(records)

    def fine_tune(self, failures_path: str, epochs: int = 1, batch_size: int = 4, use_lora: bool = True):
        if AutoTokenizer is None:
            raise RuntimeError("transformers/datasets not available in environment")
        ds = self._load_dataset(failures_path)
        if ds is None:
            print("No failures found to fine-tune on.")
            return None

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def preprocess(examples):
            # Combine input and target into single sequences for language model training
            texts = [f"{inp}\n{tgt}{tokenizer.eos_token}" for inp, tgt in zip(examples["input"], examples["target"])]
            encodings = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
            # Labels are the same as input_ids for causal language modeling
            encodings["labels"] = encodings["input_ids"].copy()
            return encodings

        tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

        model = AutoModelForCausalLM.from_pretrained(self.hf_model_id)
        
        # Apply LoRA if available and requested
        if use_lora and PEFT_AVAILABLE:
            print("Applying LoRA configuration...")
            # For GPT-Neo, use 'c_attn' and 'c_proj' from attention layers
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],  # GPT-Neo attention and projection layers
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=10,
            save_total_limit=2,
            fp16=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized)
        trainer.train()
        trainer.save_model(self.output_dir)
        return self.output_dir
