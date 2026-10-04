"""RAG answer orchestration, citation resolution, and timing breakdown."""

from __future__ import annotations

import re
import time
from typing import Any

from teleassist.config import load_config
from teleassist.generation.llm import LocalLLM
from teleassist.generation.prompts import ABSTENTION, build_messages
from teleassist.retrieval.pipeline import Retriever

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class RAGAnswerer:
    """Retrieve context and generate a cited answer, or run a plain-model baseline."""

    def __init__(
        self,
        retriever: Any | None = None,
        llm: Any | None = None,
        *,
        config_path: str = "configs/default.yaml",
        chunks_path: str = "data/processed/chunks.jsonl",
        index_path: str = "data/processed/dense_index.npz",
        adapter_path: str | None = None,
    ):
        self.config = load_config(config_path)
        self._config_path = str(config_path)
        self.retriever = retriever
        self.llm = llm or LocalLLM(
            self.config.models.generator,
            max_new_tokens=self.config.models.max_new_tokens,
            load_in_4bit=self.config.models.load_in_4bit,
            adapter_path=adapter_path,
        )
        self.chunks_path = chunks_path
        self.index_path = index_path

    def answer(
        self,
        question: str,
        mode: str = "hybrid_rerank",
        use_lora: bool = False,
    ) -> dict[str, Any]:
        """Return answer text, mapped citations, retrieval records, and elapsed times."""
        retrieval_ms = 0.0
        retrieved: list[dict[str, Any]] = []
        use_context = mode != "no_retrieval"
        if use_context:
            if self.retriever is None:
                self.retriever = Retriever.from_files(
                    self.chunks_path,
                    dense_index_path=self.index_path,
                    config_path=self._config_path,
                )
            start = time.perf_counter()
            retrieved = self.retriever.search(question, mode=mode, top_k=5)
            retrieval_ms = (time.perf_counter() - start) * 1000
        messages = build_messages(question, retrieved, use_context=use_context)
        start = time.perf_counter()
        answer = self.llm.generate(messages, use_lora=use_lora)
        generation_ms = (time.perf_counter() - start) * 1000

        citations = []
        invalid_numbers = []
        seen_numbers = set()
        for match in CITATION_PATTERN.finditer(answer):
            number = int(match.group(1))
            if number in seen_numbers:
                continue
            seen_numbers.add(number)
            if number < 1 or number > len(retrieved):
                invalid_numbers.append(number)
                continue
            chunk = retrieved[number - 1]
            metadata = chunk.get("metadata", {})
            citations.append(
                {
                    "id": chunk.get("id"),
                    "doc_title": metadata.get("doc_title"),
                    "year": metadata.get("year"),
                    "family": metadata.get("family"),
                    "snippet": " ".join(str(chunk.get("text", "")).split())[:200],
                }
            )
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved,
            "invalid_citation_numbers": invalid_numbers,
            "abstained": answer.strip() == ABSTENTION,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 3),
                "generation": round(generation_ms, 3),
                "total": round(retrieval_ms + generation_ms, 3),
            },
        }


def main() -> None:
    """Answer one question from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_rerank", "no_retrieval"],
        default="hybrid_rerank",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--index", default="data/processed/dense_index.npz")
    args = parser.parse_args()
    result = RAGAnswerer(
        config_path=args.config, chunks_path=args.chunks, index_path=args.index
    ).answer(args.query, mode=args.mode)
    print(result["answer"])
    print("Cited sources:")
    for source in result["citations"]:
        print(
            f"- [{source['id']}] {source['doc_title']} ({source['year']}); "
            f"family={source['family']} — {source['snippet']}"
        )
    if result["invalid_citation_numbers"]:
        print(f"Invalid citation numbers: {result['invalid_citation_numbers']}")
    print(f"Abstained: {result['abstained']}")
    latency = result["latency_ms"]
    print(
        f"Latency: retrieval {latency['retrieval']:.1f} ms; "
        f"generation {latency['generation']:.1f} ms; total {latency['total']:.1f} ms"
    )


if __name__ == "__main__":
    main()

