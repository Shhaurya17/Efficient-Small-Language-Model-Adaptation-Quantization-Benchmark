# Efficient Small Language Model Adaptation & Quantization Benchmark

A systematic study of QLoRA fine-tuning and 4-bit quantization for small language models.
Measures the quality-efficiency trade-offs across **MMLU, ARC, GSM8K, HellaSwag**.

> Status: repository scaffold only. Results below will be filled in after training,
> quantization, and evaluation are run (see `reports/results.md`).

## Research Questions

- **RQ1**: How much does LoRA rank affect SFT quality vs trainable parameters?
- **RQ2**: What quality is lost in 4-bit quantization?
- **RQ3**: What is the efficiency gain (VRAM/latency) per quality point?
- **RQ4**: How does inference throughput scale across checkpoints?

## Key Results

_To be filled in after Phase 7/8 (evaluation + analysis) — real Qwen2.5-1.5B-Instruct
run on GPU. The table below is a CPU pilot on a tiny proxy model, not this experiment._

| Checkpoint | MMLU | ARC | GSM8K | HellaSwag | VRAM | Throughput |
|-----------|------|-----|-------|-----------|------|-----------|
| Base (Qwen2.5-1.5B) | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT-R8 | TBD | TBD | TBD | TBD | TBD | TBD |
| 4-bit SFT-R8 | TBD | TBD | TBD | TBD | TBD | TBD |

### CPU pilot (real numbers, from a scaled-down run — not the real experiment)

These are actual measured results from `scripts/pilot_cpu_run.py`
(`eval/results/pilot_cpu/*.json`), not fabricated figures. What makes this a
"pilot" rather than the real experiment is the *setup*: no GPU is available on
the dev machine, so it substitutes HuggingFaceTB/SmolLM2-135M-Instruct for
Qwen2.5-1.5B-Instruct, int8 dynamic quantization for GPTQ, and a
10-step/80-example training run, just to exercise the pipeline end-to-end.
Full deviations, a lm-eval/quantization incompatibility hit along the way, and
two real bugs found in the configs are documented in
[`reports/pilot_cpu_findings.md`](reports/pilot_cpu_findings.md).

| Checkpoint | ARC-Easy (0-shot, n=15) | HellaSwag (0-shot, n=15) | Model size | Process RAM | Latency | Throughput |
|-----------|------|-----------|------|------|---------|-----------|
| Base (SmolLM2-135M) | 60.0% | 33.3% | 538 MB | 1.16 GB | 119.4 ms/token | 13.6 tok/s |
| SFT-R8 (merged, fp32) | 60.0% | 33.3% | 542 MB | 1.31 GB | 76.9 ms/token | 18.3 tok/s |
| Int8-quantized R8 | n/a* | n/a* | 252 MB (-53%) | 1.96 GB | 92.7 ms/token | 16.4 tok/s |

\* lm-eval's weight-tying step is incompatible with `torch.quantization.quantize_dynamic`
output; this is specific to the CPU stand-in, not the GPTQ path. See the findings doc.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python data/scripts/download_data.py

# 3. Run training
python -m efficient_slm.training.trainer --config configs/train.yaml --lora_rank 8

# 4. Merge + quantize model
python -m efficient_slm.quantization.merge --lora_rank 8
python -m efficient_slm.quantization.gptq --model outputs/merged_r8 --output outputs/quantized_r8

# 5. Evaluate
python -m efficient_slm.evaluation.benchmark --model outputs/quantized_r8 --benchmarks mmlu arc gsm8k hellaswag
```

## Repository Structure

- `configs/`: YAML configurations for all experiments
- `data/`: Data loading and preprocessing
- `src/efficient_slm/`: Training, quantization, inference, evaluation modules
- `notebooks/`: End-to-end workflows (00-07)
- `eval/`: Benchmark results
- `reports/`: Analysis and findings
- `figures/`: Visualizations

## Hardware & Compute

- GPU: NVIDIA T4 (16GB VRAM) or A100 (Colab)
- Max training budget: 24 GPU hours total
- Inference: single GPU sufficient for all benchmarks

## Citation

If you use this project, please cite:

```bibtex
@misc{efficient-slm-benchmark,
  title={Efficient Small Language Model Adaptation & Quantization Benchmark},
  author={[Your Name]},
  year={2026},
  url={https://github.com/Shhaurya17/Efficient-Small-Language-Model-Adaptation-Quantization-Benchmark}
}
```
