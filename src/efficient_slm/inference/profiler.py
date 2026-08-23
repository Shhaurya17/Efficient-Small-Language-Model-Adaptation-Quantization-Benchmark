import time

from efficient_slm.inference.engine import load_model, measure_latency, measure_throughput, measure_vram


def profile_checkpoint(model_path, torch_dtype="float16"):
    start = time.perf_counter()
    model, tokenizer = load_model(model_path, quantized=False, torch_dtype=torch_dtype)
    load_time_sec = time.perf_counter() - start

    vram = measure_vram(model)
    latency = measure_latency(model, tokenizer, batch_size=1, seq_length=256, num_samples=100)
    throughput = measure_throughput(model, tokenizer, batch_size=4, seq_length=512)

    report = {
        "model_path": model_path,
        "load_time_sec": load_time_sec,
        "vram": vram,
        "latency": latency,
        "throughput": throughput,
    }
    return report, model, tokenizer


def generate_report(all_measurements):
    return all_measurements
