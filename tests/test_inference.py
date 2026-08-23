import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from efficient_slm.inference.engine import generate, load_model, measure_latency, measure_vram

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model(TINY_MODEL, torch_dtype="float32", device_map="cpu")


def test_load_model_returns_model_and_tokenizer(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None
    assert model.device.type == "cpu"


def test_generate_returns_nonempty_string(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    output = generate(model, tokenizer, "Hello", max_new_tokens=5)
    assert isinstance(output, str)


def test_measure_vram_returns_expected_shape(model_and_tokenizer):
    model, _ = model_and_tokenizer
    vram = measure_vram(model)
    assert set(vram.keys()) == {"peak_vram_gb", "allocated_vram_gb"}
    if torch.cuda.is_available():
        assert isinstance(vram["peak_vram_gb"], float)
    else:
        assert vram["peak_vram_gb"] is None
        assert vram["allocated_vram_gb"] is None


def test_measure_latency_returns_expected_keys(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    result = measure_latency(model, tokenizer, batch_size=1, num_samples=2)
    assert set(result.keys()) == {"batch_size", "ms_per_token", "throughput_tokens_per_sec"}
    assert result["batch_size"] == 1
    assert result["ms_per_token"] > 0
    assert result["throughput_tokens_per_sec"] > 0


def test_measure_latency_scales_with_batch_size(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    result = measure_latency(model, tokenizer, batch_size=4, num_samples=2)
    assert result["batch_size"] == 4
