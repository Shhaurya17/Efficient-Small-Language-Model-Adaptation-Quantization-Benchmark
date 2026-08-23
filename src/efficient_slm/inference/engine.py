import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name, quantized=False, torch_dtype="float16", device_map="auto"):
    dtype = getattr(torch, torch_dtype)

    quantization_config = None
    if quantized:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def measure_vram(model):
    if model.device.type != "cuda":
        return {"peak_vram_gb": None, "allocated_vram_gb": None}
    torch.cuda.synchronize()
    return {
        "peak_vram_gb": torch.cuda.max_memory_allocated(model.device) / 1e9,
        "allocated_vram_gb": torch.cuda.memory_allocated(model.device) / 1e9,
    }


def measure_latency(model, tokenizer, batch_size=1, seq_length=256, num_samples=20):
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", truncation=True, max_length=seq_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize()

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)  # warmup

    start = time.perf_counter()
    generated_tokens = 0
    with torch.no_grad():
        for _ in range(num_samples):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_tokens += output_ids.shape[1] - inputs["input_ids"].shape[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "ms_per_token": (elapsed / generated_tokens) * 1000,
        "throughput_tokens_per_sec": generated_tokens / elapsed,
    }


def measure_throughput(model, tokenizer, batch_size=4, seq_length=512, num_samples=5):
    return measure_latency(model, tokenizer, batch_size=batch_size, seq_length=seq_length, num_samples=num_samples)


def generate_report(measurements):
    return {
        "vram": measurements.get("vram"),
        "latency": measurements.get("latency"),
        "throughput": measurements.get("throughput"),
    }
