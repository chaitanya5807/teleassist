"""Supervised LoRA fine-tuning with assistant-completion-only loss."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from teleassist.generation.prompts import build_messages


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/teleassist-lora"))
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model_name = (
        cfg.get("smoke_model_name")
        if args.smoke
        else cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    )
    train_rows = _read(args.data_dir / "train.jsonl")
    val_rows = _read(args.data_dir / "val.jsonl")
    if args.smoke:
        train_rows, val_rows = (train_rows + val_rows)[:32], (train_rows + val_rows)[:8]
    if not train_rows or not val_rows:
        raise ValueError("Training requires non-empty train.jsonl and val.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def format_row(row: dict[str, Any]) -> dict[str, str]:
        messages = build_messages(row["question"], row.get("context", []))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "completion": row["answer"] + tokenizer.eos_token}

    quant = None
    if torch.cuda.is_available() and cfg.get("quantization") == "nf4":
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto" if quant else None,
        torch_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    peft = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(args.checkpoint_dir),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["epochs"]),
        lr_scheduler_type=cfg["schedule"],
        per_device_train_batch_size=1 if args.smoke else int(cfg.get("train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 2)),
        gradient_accumulation_steps=1
        if args.smoke
        else int(cfg.get("gradient_accumulation_steps", 8)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        dataset_text_field="text",
        completion_only_loss=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1 if args.smoke else 10,
        max_steps=20 if args.smoke else -1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=42,
    )

    class LossCSV(TrainerCallback):
        def __init__(self) -> None:
            self.path = Path("results/train_log.csv")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=["step", "train_loss", "val_loss"])
            self.writer.writeheader()

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            logs = logs or {}
            self.writer.writerow(
                {
                    "step": state.global_step,
                    "train_loss": logs.get("loss", ""),
                    "val_loss": logs.get("eval_loss", ""),
                }
            )
            self.file.flush()
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.file.close()
            return control

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list([format_row(row) for row in train_rows]),
        eval_dataset=Dataset.from_list([format_row(row) for row in val_rows]),
        peft_config=peft,
        processing_class=tokenizer,
        callbacks=[LossCSV()],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    import matplotlib.pyplot as plt

    with Path("results/train_log.csv").open(encoding="utf-8") as stream:
        log = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in log]
    for key, label in (("train_loss", "train"), ("val_loss", "validation")):
        pairs = [(step, float(row[key])) for step, row in zip(steps, log, strict=True) if row[key]]
        if pairs:
            plt.plot([x[0] for x in pairs], [x[1] for x in pairs], label=label)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
