import pytest

from teleassist.evaluation.build_eval_set import build_eval_set, verify_unanswerable
from teleassist.generation.question_generator import (
    MockQuestionGenerator,
    OpenAICompatibleGenerator,
    _verify_evidence,
)
from teleassist.retrieval.bm25 import BM25Index
from teleassist.training.build_sft_data import (
    annotate_qa_quality,
    make_sft_record,
)


def make_chunk(identifier, source, family, text):
    return {
        "id": identifier,
        "text": text,
        "metadata": {"source": source, "family": family, "doc_title": family, "year": 2024},
    }


def test_quality_filter_flags_types_without_removing_chunks():
    chunks = [
        make_chunk("toc", "doc-a", "AIRTEL", "Contents\nBilling ........ 3\nRoaming ........ 8"),
        make_chunk(
            "contact",
            "doc-a",
            "AIRTEL",
            "Contact address: 12 Main Road\nPhone: 18001234567\nEmail: care@example.in\n"
            "Office address: 5 City Road\nContact: 121\nPhone: 18005550123\n"
            "Email: another@example.in\n" + "customer service details " * 20,
        ),
        make_chunk("short", "doc-a", "AIRTEL", "A short chunk with only a few words."),
        make_chunk(
            "numbers",
            "doc-a",
            "AIRTEL",
            ("123 456 789 012\n" * 20) + "table values " * 30,
        ),
        make_chunk(
            "qos",
            "doc-b",
            "QOS",
            "Input of Stakeholders: operators submitted views about this measure. " * 10,
        ),
        make_chunk(
            "good",
            "doc-a",
            "AIRTEL",
            "Customers can ask the operator to explain the monthly bill and report an error "
            "through the complaint centre, by phone, or by visiting an authorised service office. "
            * 3,
        ),
    ]

    counts = annotate_qa_quality(chunks)

    assert len(chunks) == 6
    assert [chunk["usable_for_qa"] for chunk in chunks] == [False] * 5 + [True]
    assert counts == {
        "table_of_contents_or_index": 1,
        "address_or_contact_list": 1,
        "under_40_words": 1,
        "mostly_numbers_or_tables": 1,
        "qos_stakeholder_discussion": 1,
    }


def test_generator_rejects_evidence_quotes_not_present():
    items = _verify_evidence(
        [
            {"question": "Q?", "answer": "A [1]", "evidence": "Exact text"},
            {"question": "Q?", "answer": "A [1]", "evidence": "Missing text"},
        ],
        "Exact text is present.",
    )

    assert len(items) == 1
    assert items[0]["evidence"] == ["Exact text"]
    assert MockQuestionGenerator().generate("A valid passage text.", 2)[0]["evidence"]


def test_sft_record_rewrites_citation_after_shuffling_context():
    gold = make_chunk("gold", "train.pdf", "AIRTEL", "Customers can port their mobile number.")
    negatives = [
        make_chunk(f"neg-{i}", "train.pdf", "AIRTEL", f"Other unrelated policy text {i}.")
        for i in range(3)
    ]
    import random

    record = make_sft_record(
        "How do I port my number?",
        "Customers can port their mobile number [1].",
        ["Customers can port their mobile number."],
        gold,
        negatives,
        answerable=True,
        rng=random.Random(42),
    )

    assert record is not None
    assert len(record["context"]) == 4
    gold_position = next(i for i, ctx in enumerate(record["context"], 1) if ctx["id"] == "gold")
    assert record["answer"].endswith(f"[{gold_position}]")


def test_sft_record_supports_five_passages_and_gold_last():
    import random

    gold = make_chunk("gold", "train.pdf", "AIRTEL", "Customers can port their mobile number.")
    negatives = [
        make_chunk(f"neg-{i}", "train.pdf", "AIRTEL", f"Other unrelated policy text {i}.")
        for i in range(4)
    ]
    record = make_sft_record(
        "How do I port my number?",
        "Customers can port their mobile number [1].",
        ["Customers can port their mobile number."],
        gold,
        negatives,
        answerable=True,
        rng=random.Random(42),
        five_context=True,
        gold_last=True,
    )

    assert record is not None
    assert len(record["context"]) == 5
    assert record["context"][-1]["id"] == "gold"
    assert record["answer"].endswith("[5]")


def test_openai_compatible_generator_uses_environment_and_verifies_evidence(
    monkeypatch,
):
    import teleassist.generation.question_generator as generator_module

    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "test-model")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"question":"How?","answer":"Text [1]",'
                            '"evidence":"supporting text"}]'
                        }
                    }
                ]
            }

    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(generator_module.requests, "post", fake_post)
    result = OpenAICompatibleGenerator().generate("A passage with supporting text.", 1)

    assert result[0]["evidence"] == ["supporting text"]
    assert calls[0][0] == "https://example.test/v1/chat/completions"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0][1]["json"]["model"] == "test-model"


def test_eval_builder_uses_only_eval_sources_and_has_multi_unanswerable_items():
    topics = [
        "airtime",
        "broadband",
        "roaming",
        "voucher",
        "recharge",
        "invoice",
        "porting",
        "service",
        "network",
        "complaint",
        "handset",
        "coverage",
        "billing",
        "tariff",
        "subscriber",
        "mobile",
        "wireless",
        "prepaid",
        "postpaid",
        "operator",
        "customer",
        "quality",
        "number",
        "account",
    ]
    chunks = [
        make_chunk(
            f"eval-{index}",
            "eval-a.pdf" if index < 12 else "eval-b.pdf",
            "AIRTEL",
            f"The customer receives information about {topic} from the telecom provider. "
            "Consumers may request assistance from the service centre and review the terms "
            "before selecting a plan. This passage describes a practical customer service rule "
            "and provides useful guidance for making an informed choice.",
        )
        for index, topic in enumerate(topics)
    ]
    chunks += [
        make_chunk(
            "train-1",
            "train.pdf",
            "JIO",
            "Train only material describes a separate service process for subscribers.",
        )
    ]
    split = {
        "eval_documents": [
            {"source": "eval-a.pdf", "family": "AIRTEL"},
            {"source": "eval-b.pdf", "family": "AIRTEL"},
        ],
        "train_documents": [{"source": "train.pdf", "family": "JIO"}],
    }

    records = build_eval_set(
        chunks,
        split,
        MockQuestionGenerator(),
        target_count=8,
        seed=2,
        max_doc_share=1.0,
    )

    assert len(records) == 8
    assert sum(not record["answerable"] for record in records) == 2
    assert any(len(record["gold_chunk_ids"]) == 2 for record in records)
    assert all(record["verified"] is False for record in records)
    assert all(
        chunk_id.startswith("eval-") for record in records for chunk_id in record["gold_chunk_ids"]
    )
    assert verify_unanswerable(
        "Can I transfer bank account credit?", "rare unsupported phrase", BM25Index(chunks)
    )


def test_eval_build_dedupes_questions_and_rejects_missing_split_source():
    chunks = [make_chunk("not-eval", "train.pdf", "AIRTEL", "Customer service text " * 20)]
    with pytest.raises(ValueError, match="No usable QA chunks"):
        build_eval_set(
            chunks,
            {"eval_documents": [], "train_documents": []},
            MockQuestionGenerator(),
            target_count=10,
        )
