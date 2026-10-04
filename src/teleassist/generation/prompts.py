"""Chat prompts for grounded RAG and the no-retrieval baseline."""

from __future__ import annotations

from typing import Any

ABSTENTION = "I could not find this in the provided documents."
SYSTEM_PROMPT = f"""You are TeleAssist, a telecom support assistant.
Answer ONLY from the numbered context passages supplied by the user. Cite each factual
claim with its passage number in square brackets, such as [1] or [2]. Keep answers concise.
If the passages do not contain the answer, reply exactly: "{ABSTENTION}"""
BASELINE_SYSTEM_PROMPT = (
    "You are TeleAssist, a telecom support assistant. Answer the user's question directly and "
    "concisely. Do not claim to have consulted documents or cite sources."
)


def build_messages(
    question: str,
    context: list[dict[str, Any]] | None = None,
    *,
    use_context: bool = True,
) -> list[dict[str, str]]:
    """Format numbered context and question as messages for the model chat template."""
    if not use_context:
        return [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ]
    passages = []
    for number, chunk in enumerate(context or [], start=1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("doc_title") or "Unknown title"
        year = metadata.get("year")
        year_text = str(year) if year is not None else "year unknown"
        passages.append(
            f"[{number}] {title} ({year_text})\n{str(chunk.get('text', '')).strip()}"
        )
    context_text = "\n\n".join(passages) if passages else "(No context passages were retrieved.)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n{context_text}\n\nQuestion: {question.strip()}",
        },
    ]
