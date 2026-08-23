import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efficient_slm.data.loader import (
    compute_stats,
    filter_valid_pairs,
    format_pair,
    remove_duplicates,
    save_to_disk,
    train_val_split,
)


def test_filter_valid_pairs_drops_short_empty_and_missing():
    pairs = [
        {"instruction": "a", "response": "too short"},
        {"instruction": "b", "response": " ".join(["word"] * 20)},
        {"instruction": "c", "response": ""},
        {"instruction": "", "response": "non-empty but orphaned"},
    ]
    result = filter_valid_pairs(pairs, min_length=10, max_length=2048)
    assert result == [{"instruction": "b", "response": " ".join(["word"] * 20)}]


def test_filter_valid_pairs_drops_over_max_length():
    pairs = [{"instruction": "a", "response": " ".join(["word"] * 3000)}]
    assert filter_valid_pairs(pairs, min_length=10, max_length=2048) == []


def test_remove_duplicates_keeps_first_occurrence():
    pairs = [
        {"instruction": "a", "response": "b"},
        {"instruction": "a", "response": "b"},
        {"instruction": "a", "response": "c"},
    ]
    result = remove_duplicates(pairs)
    assert result == [{"instruction": "a", "response": "b"}, {"instruction": "a", "response": "c"}]


def test_train_val_split_ratio_and_no_overlap():
    pairs = [{"instruction": str(i), "response": str(i)} for i in range(100)]
    train, val = train_val_split(pairs, ratio=0.8, seed=42)
    assert len(train) == 80
    assert len(val) == 20
    train_ids = {p["instruction"] for p in train}
    val_ids = {p["instruction"] for p in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {str(i) for i in range(100)}


def test_train_val_split_is_deterministic_for_same_seed():
    pairs = [{"instruction": str(i), "response": str(i)} for i in range(20)]
    train1, val1 = train_val_split(pairs, ratio=0.8, seed=42)
    train2, val2 = train_val_split(pairs, ratio=0.8, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_format_pair_alpaca_template():
    pair = {"instruction": "What is 2+2?", "response": "4"}
    formatted = format_pair(pair, "alpaca")
    assert "### Instruction:" in formatted
    assert "What is 2+2?" in formatted
    assert "4" in formatted


def test_format_pair_default_template():
    pair = {"instruction": "Hi", "response": "Hello"}
    formatted = format_pair(pair, "default")
    assert formatted == "[INST] Hi [/INST] Hello"


def test_compute_stats_counts_and_averages():
    train = [{"instruction": "hello world", "response": "hi there friend"}]
    val = [{"instruction": "a b", "response": "c d e"}]
    stats = compute_stats(train, val, chat_template="alpaca")
    assert stats["train_pairs"] == 1
    assert stats["val_pairs"] == 1
    assert stats["total_pairs"] == 2
    assert stats["avg_tokens"] > 0
    assert stats["max_tokens"] >= stats["min_tokens"]


def test_save_to_disk_writes_jsonl(tmp_path):
    pairs = [{"instruction": "a", "response": "b"}, {"instruction": "c", "response": "d"}]
    out_path = tmp_path / "nested" / "out.jsonl"
    save_to_disk(pairs, out_path)

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == pairs
