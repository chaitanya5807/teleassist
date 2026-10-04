"""Lazy Hugging Face causal language model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class LocalLLM:
    """Load a chat model on first use and generate with deterministic decoding."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 256,
        load_in_4bit: bool = True,
        adapter_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._loaded_with_lora = False

    def _load(self, *, use_lora: bool) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"device_map": "auto" if device == "cuda" else None}
        if device == "cuda" and self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if device == "cpu":
            model.to("cpu")
        if use_lora:
            if self.adapter_path is None:
                raise ValueError("use_lora=True requires an adapter_path")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        if hasattr(model, "generation_config"):
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        self._model = model
        self._loaded_with_lora = use_lora

    def generate(
        self, messages: list[dict[str, str]], *, use_lora: bool = False
    ) -> str:
        """Apply the model's chat template and return only newly generated text."""
        import torch

        if self._model is None or self._tokenizer is None or self._loaded_with_lora != use_lora:
            self._load(use_lora=use_lora)
        elif use_lora and self.adapter_path is None:
            raise ValueError("use_lora=True requires an adapter_path")
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
