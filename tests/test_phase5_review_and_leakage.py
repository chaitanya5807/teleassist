import json

from scripts.check_data_leakage import check_leakage
from scripts.review_eval_set import review

from teleassist.resumable import append_resumable


def test_review_tool_edits_and_saves_each_item(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    item = {
        "id": "eval-0001",
        "question": "Old question?",
        "reference_answer": "Old answer.",
        "evidence": ["evidence quote"],
        "gold_chunk_ids": ["gold"],
        "verified": False,
    }
    eval_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    chunks_path.write_text(
        json.dumps({"id": "gold", "text": "gold chunk text"}) + "\n", encoding="utf-8"
    )
    answers = iter(["e", "q", "New question?", "y"])

    review(eval_path, chunks_path, input_fn=lambda _: next(answers), print_fn=lambda _: None)

    saved = json.loads(eval_path.read_text(encoding="utf-8"))
    assert saved["question"] == "New question?"
    assert saved["reference_answer"] == "Old answer."
    assert saved["verified"] is True


def test_leakage_check_allows_train_gold_and_reports_similarity(capsys):
    chunks = [
        {"id": "train-id", "metadata": {"source": "train.pdf", "family": "AIRTEL"}},
        {"id": "eval-id", "metadata": {"source": "eval.pdf", "family": "MNP"}},
    ]
    train = [
        {
            "question": "How can I contact customer care?",
            "gold_chunk_ids": ["train-id"],
            "context": [{"id": "train-id"}],
        }
    ]
    eval_items = [{"question": "How do I reach customer care?"}]
    split = {
        "eval_documents": [{"source": "eval.pdf", "family": "MNP"}],
        "train_documents": [{"source": "train.pdf", "family": "AIRTEL"}],
    }

    similarity = check_leakage(train, eval_items, chunks, split)

    assert 0 <= similarity <= 1
    assert "Maximum eval/train question similarity:" in capsys.readouterr().out


def test_leakage_check_rejects_gold_from_eval_family():
    chunks = [{"id": "eval-id", "metadata": {"source": "eval.pdf", "family": "MNP"}}]
    train = [{"question": "Question?", "gold_chunk_ids": ["eval-id"], "context": []}]
    split = {
        "eval_documents": [{"source": "eval.pdf", "family": "MNP"}],
        "train_documents": [],
    }

    try:
        check_leakage(train, [], chunks, split)
    except AssertionError as error:
        assert "belongs to eval data" in str(error)
    else:
        raise AssertionError("leakage check accepted an eval gold chunk")


def test_resumable_mock_generation_stops_and_resumes_without_duplicates(tmp_path):
    chunks = [{"id": f"c{i}"} for i in range(10)]
    records = [{"id": f"r{i}", "chunk_id": f"c{i}"} for i in range(10)]
    append_resumable(
        chunks,
        records,
        out_dir=tmp_path,
        max_chunks=5,
        chunk_ids=lambda row: [row["chunk_id"]],
        filename="items.jsonl",
    )
    assert len((tmp_path / "items.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    append_resumable(
        chunks,
        records,
        out_dir=tmp_path,
        max_chunks=None,
        chunk_ids=lambda row: [row["chunk_id"]],
        filename="items.jsonl",
    )
    lines = (tmp_path / "items.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert len({json.loads(line)["id"] for line in lines}) == 10
