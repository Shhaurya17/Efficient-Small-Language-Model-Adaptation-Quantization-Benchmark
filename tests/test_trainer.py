import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from transformers import AutoTokenizer

from efficient_slm.training.trainer import build_lora_config, tokenize_dataset

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_build_lora_config_sets_expected_fields():
    config = build_lora_config(rank=8, alpha=16, dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none")
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) == {"q_proj", "v_proj"}
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"


def test_tokenize_dataset_produces_input_ids(tokenizer):
    pairs = [{"instruction": "What is 2+2?", "response": "4"}]
    dataset = tokenize_dataset(pairs, tokenizer, chat_template="alpaca", max_seq_length=64)
    assert len(dataset) == 1
    assert "input_ids" in dataset.column_names
    assert len(dataset[0]["input_ids"]) > 0


def test_tokenize_dataset_respects_max_seq_length(tokenizer):
    long_pair = [{"instruction": "hi", "response": " ".join(["word"] * 500)}]
    dataset = tokenize_dataset(long_pair, tokenizer, chat_template="alpaca", max_seq_length=32)
    assert len(dataset[0]["input_ids"]) <= 32


def test_tokenize_dataset_handles_multiple_pairs(tokenizer):
    pairs = [
        {"instruction": "Hi", "response": "Hello"},
        {"instruction": "Bye", "response": "Goodbye"},
    ]
    dataset = tokenize_dataset(pairs, tokenizer, chat_template="default", max_seq_length=64)
    assert len(dataset) == 2
