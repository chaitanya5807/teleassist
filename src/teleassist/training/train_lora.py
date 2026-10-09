"""Supervised LoRA fine-tuning (TRL 0.13) with answer-only loss."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from teleassist.generation.prompts import build_messages
from teleassist.generation.question_generator import create_generator

DEFAULT_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _filter_by_length(
    rows: list[dict[str, str]], tokenizer: Any, max_len: int, name: str
) -> list[dict[str, str]]:
    """Drop examples longer than max_len so the answer is never truncated away."""
    kept = []
    for row in rows:
        n_tokens = len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        if n_tokens <= max_len:
            kept.append(row)
    dropped = len(rows) - len(kept)
    print(f"[{name}] kept {len(kept)} of {len(rows)} (dropped {dropped} over {max_len} tokens)")
    if not kept:
        raise ValueError(f"No {name} examples fit in max_seq_length={max_len}")
    return kept


def _check_masking(collator: Any, tokenizer: Any, texts: list[str]) -> None:
    """Fail early if the loss mask leaves no answer tokens to learn from."""
    for i, text in enumerate(texts[:8]):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        labels = collator([{"input_ids": ids}])["labels"][0].tolist()
        trained = [t for t in labels if t != -100]
        if not trained:
            raise ValueError(f"Example {i}: response template not found, every label is masked")
        if i == 0:
            print("Loss is computed ONLY on this text:\n" + tokenizer.decode(trained) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/teleassist-lora"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    model_name = (
        cfg.get("smoke_model_name")
        if args.smoke
        else cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    )
    if args.smoke:
        mock = create_generator("mock")
        text = (
            "Telecom customers can contact support to ask about recharge, billing, "
            "and mobile network services."
        )
        sample = mock.generate(text, 1)[0]
        context = [{"id": f"mock-{i}", "text": text} for i in range(3)]
        train_rows = [
            dict(sample, question=f"{sample['question']} Example {i}", context=context)
            for i in range(24)
        ]
        val_rows = [
            dict(sample, question=f"{sample['question']} Validation {i}", context=context)
            for i in range(8)
        ]
    else:
        train_rows = _read(args.data_dir / "train.jsonl")
        val_rows = _read(args.data_dir / "val.jsonl")
    if not train_rows or not val_rows:
        raise ValueError("Training requires non-empty train.jsonl and val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.smoke)
    max_len = int(cfg.get("max_seq_length", 2048))

    def format_row(row: dict[str, Any]) -> dict[str, str]:
        messages = build_messages(row["question"], row.get("context", []))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"text": prompt + row["answer"] + tokenizer.eos_token}

    train_text = _filter_by_length([format_row(r) for r in train_rows], tokenizer, max_len, "train")
    val_text = _filter_by_length([format_row(r) for r in val_rows], tokenizer, max_len, "val")

    template = cfg.get("response_template", DEFAULT_RESPONSE_TEMPLATE)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(template, add_special_tokens=False),
        tokenizer=tokenizer,
    )
    _check_masking(collator, tokenizer, [r["text"] for r in train_text])

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
        local_files_only=args.smoke,
        quantization_config=quant,
        device_map="auto" if quant else None,
        torch_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32),
    )
    model.config.use_cache = False
    peft = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    eval_every = 10 if args.smoke else int(cfg.get("eval_steps", 50))
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
        max_seq_length=max_len,
        dataset_text_field="text",
        dataset_kwargs={"add_special_tokens": False},
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="no" if args.smoke else "steps",
        save_steps=eval_every,
        save_total_limit=2,
        logging_steps=1 if args.smoke else 10,
        max_steps=20 if args.smoke else -1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=42,
    )

    log_path = args.results_dir / "train_log.csv"
    if args.resume_from_checkpoint is None and log_path.exists():
        log_path.unlink()

    class LossCSV(TrainerCallback):
        def __init__(self) -> None:
            is_new = not log_path.exists() or log_path.stat().st_size == 0
            self.file = log_path.open("a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=["step", "train_loss", "val_loss"])
            if is_new:
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
            if "loss" not in logs and "eval_loss" not in logs:
                return control
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
        train_dataset=Dataset.from_list(train_text),
        eval_dataset=Dataset.from_list(val_text),
        peft_config=peft,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[LossCSV()],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)

    import matplotlib.pyplot as plt

    with log_path.open(encoding="utf-8") as stream:
        log = list(csv.DictReader(stream))
    for key, label in (("train_loss", "train"), ("val_loss", "validation")):
        pairs = [(int(r["step"]), float(r[key])) for r in log if r[key]]
        if pairs:
            plt.plot([p[0] for p in pairs], [p[1] for p in pairs], label=label)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.results_dir / "loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()