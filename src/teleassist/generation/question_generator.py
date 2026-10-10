"""Question and answer generator backends with evidence quote verification."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Any, Protocol

import requests

from teleassist.generation.llm import LocalLLM

QUESTION_SYSTEM = """Create realistic questions a telecom customer might ask, in plain everyday
wording. Do not copy distinctive phrases from the passage into the question. Answer using ONLY
the supplied passage(s), cite the supporting passage as [1] (and [2] when both passages are
needed), and include one short exact quote as evidence. Return only a JSON array of objects with
question, answer, and evidence fields. Every evidence quote must occur verbatim in its passage."""


class QuestionGenerator(Protocol):
    """Common question generation interface."""

    def generate(self, chunk_text: str, n: int) -> list[dict[str, Any]]: ...


def _parse_items(response: str) -> list[dict[str, Any]]:
    """Read the first valid JSON list of objects; ignore any text around or after it."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", response):
        try:
            data, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and any(isinstance(item, dict) for item in data):
            return [item for item in data if isinstance(item, dict)]
    raise ValueError("Question generator did not return a JSON array of objects")


def _verify_evidence(items: list[dict[str, Any]], chunk_text: str) -> list[dict[str, Any]]:
    normalized = re.sub(r"\s+", " ", chunk_text).casefold()
    verified = []
    for item in items:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        evidence_value = item.get("evidence", "")
        evidence = [evidence_value] if isinstance(evidence_value, str) else evidence_value
        if not question or not answer or not isinstance(evidence, list) or not evidence:
            continue
        quotes = [str(quote).strip() for quote in evidence]
        if all(quote and re.sub(r"\s+", " ", quote).casefold() in normalized for quote in quotes):
            verified.append({"question": question, "answer": answer, "evidence": quotes})
    return verified


class OpenAICompatibleGenerator:
    """Call a chat-completions compatible endpoint using BASE_URL/API_KEY/MODEL."""

    def __init__(self, *, timeout: int = 300):
        self.base_url = os.environ["BASE_URL"].rstrip("/")
        self.api_key = os.environ["API_KEY"]
        self.model = os.environ["MODEL"]
        self.timeout = timeout

    def generate(self, chunk_text: str, n: int) -> list[dict[str, Any]]:
        import hashlib
        cache_path = os.environ.get("QG_CACHE", "data/qg_cache.jsonl")
        cache = getattr(OpenAICompatibleGenerator, "_cache", None)
        if cache is None:
            cache = {}
            if os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rec = json.loads(line)
                            cache[rec["key"]] = rec["items"]
            OpenAICompatibleGenerator._cache = cache
        key = hashlib.sha256(f"{self.model}|{n}|{chunk_text}".encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]
        calls = getattr(OpenAICompatibleGenerator, "_calls", 0)
        cap = int(os.environ.get("MAX_API_CALLS", "0"))
        if cap and calls >= cap:
            raise SystemExit(f"Stopped: reached MAX_API_CALLS={cap}. Answers so far are saved in {cache_path}.")
        OpenAICompatibleGenerator._calls = calls + 1
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        min_gap = float(os.environ.get("MIN_REQUEST_INTERVAL", "13"))
        wait = min_gap - (time.time() - getattr(OpenAICompatibleGenerator, "_last_call", 0.0))
        if wait > 0:
            time.sleep(wait)
        for attempt in range(5):
            OpenAICompatibleGenerator._last_call = time.time()
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.7,
                    "reasoning_effort": os.environ.get("REASONING_EFFORT", "low"),
                    "messages": [
                        {"role": "system", "content": QUESTION_SYSTEM},
                        {
                            "role": "user",
                            "content": f"Generate {n} items from this passage:\n{chunk_text}",
                        },
                    ],
                },
                timeout=self.timeout,
            )
            status_code = getattr(response, "status_code", 200)
            if status_code not in {429, 500, 502, 503, 504} or attempt == 4:
                break
            if status_code == 429 and "PerDay" in response.text:
                raise RuntimeError(f"Daily quota exhausted, stopping: {response.text[:300]}")
            time.sleep(65 if status_code == 429 else min(2**attempt, 30))
        if getattr(response, "status_code", 200) >= 400:
            raise RuntimeError(f"API error {response.status_code}: {response.text[:600]}")
        content = response.json()["choices"][0]["message"]["content"]
        try:
            result = _verify_evidence(_parse_items(content), chunk_text)[:n]
        except ValueError:
            result = []
        cache[key] = result
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "preview": chunk_text[:120], "items": result}, ensure_ascii=False) + "\n")
        return result


class LocalHFQuestionGenerator:
    """Use a local HF chat model, defaulting to Qwen2.5-7B with CUDA-only 4-bit loading."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        llm: Any | None = None,
    ):
        self.llm = llm or LocalLLM(model_name, max_new_tokens=1200, load_in_4bit=True)

    def generate(self, chunk_text: str, n: int) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": QUESTION_SYSTEM},
            {"role": "user", "content": f"Generate {n} items from this passage:\n{chunk_text}"},
        ]
        return _verify_evidence(_parse_items(self.llm.generate(messages)), chunk_text)[:n]


class MockQuestionGenerator:
    """Deterministic canned generator for tests and sample plumbing runs."""

    def generate(self, chunk_text: str, n: int) -> list[dict[str, Any]]:
        passages = re.split(r"\n\s*\[([1-9]\d*)\]\s*\n", chunk_text)
        texts = [part.strip() for part in passages if part.strip() and not part.strip().isdigit()]
        if not texts:
            return []
        snippets = [re.sub(r"\s+", " ", text)[:100].strip() for text in texts[:2]]
        multi = len(snippets) > 1
        answer = " ".join(snippets)
        answer += " [1] [2]" if multi else " [1]"
        evidence = snippets if multi else [snippets[0]]
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "your",
            "their",
            "which",
            "shall",
            "there",
            "customer",
            "customers",
            "receives",
            "information",
            "telecom",
            "provider",
            "consumers",
            "request",
            "assistance",
            "review",
            "before",
            "selecting",
            "passage",
            "describes",
            "practical",
            "service",
            "provides",
            "useful",
            "guidance",
            "making",
            "informed",
            "choice",
            "about",
            "operator",
            "operators",
        }
        topics = []
        for text in texts[:2]:
            words = [
                word.casefold()
                for word in re.findall(r"[A-Za-z]{5,}", text)
                if word.casefold() not in stopwords
            ]
            frequencies = Counter(words)
            topics.append(
                next(
                    (word for word in words if frequencies[word] == 1),
                    words[0] if words else "service",
                )
            )
        topic = " and ".join(dict.fromkeys(topics)) or "the service"
        item = {
            "question": (
                f"How do the rules about {topic} work together?"
                if multi
                else f"What should a customer know about {topic}?"
            ),
            "answer": answer,
            "evidence": evidence,
        }
        return _verify_evidence([item.copy() for _ in range(max(0, n))], chunk_text)


def create_generator(
    backend: str, *, model_name: str = "Qwen/Qwen2.5-7B-Instruct"
) -> QuestionGenerator:
    """Instantiate the selected question generation backend."""
    if backend == "openai":
        return OpenAICompatibleGenerator()
    if backend == "local_hf":
        return LocalHFQuestionGenerator(model_name)
    if backend == "mock":
        return MockQuestionGenerator()
    raise ValueError(f"Unknown question generation backend: {backend}")
