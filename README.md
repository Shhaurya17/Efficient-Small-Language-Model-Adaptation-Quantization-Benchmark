# Efficient SLM Adaptation & Quantization Benchmark

Measuring what LoRA rank and 4-bit quantization actually do to a small language model —
end to end, on one consumer GPU.

This project takes Qwen2.5-1.5B-Instruct through a full lifecycle at three adapter sizes and
measures each step independently:

```
                       ┌──▶ SFT r4  ──NF4──▶ quantized r4
  Qwen2.5-1.5B ──QLoRA─┼──▶ SFT r8  ──NF4──▶ quantized r8
     (base)            └──▶ SFT r16 ──NF4──▶ quantized r16
        │                     │                    │
        └──── eval ───────────┴──── eval ──────────┘
                (Δ SFT)            (Δ Quant)
```

The deliverable isn't a fine-tuned model — it's a 7-checkpoint comparison that separates three
effects that are usually reported tangled together: **how much does instruction tuning move the
benchmarks, how much of that depends on adapter rank, and how much quality does 4-bit
deployment give back?** Rank is a real axis here, not a fixed hyperparameter: r4, r8 and r16
are trained identically and evaluated identically, so the quality-per-trainable-parameter
curve is a measured result rather than an assumption.

Everything runs on a single 16 GB card. No Colab, no paid runtime, no multi-GPU.

## Status

The pipeline runs end to end and every stage has been executed on real hardware. All seven
checkpoints exist, and all three LoRA ranks are trained identically — same 4,000 examples, same
250 optimizer steps, same hyperparameters and seed — so the rank axis is a controlled comparison
rather than an artifact of differing run lengths.

**The efficiency axis is measured and reported.** The **quality axis is still running**: the
benchmark suite is executing across all seven checkpoints, and its table stays empty until it
finishes rather than being filled from a partial or under-sampled run. Neither Δ SFT nor
Δ Quant has a data point yet. See [Results](#results).

## Quickstart

```bash
make setup                 # deps + editable install
make data                  # download & preprocess OpenAssistant
make train                 # QLoRA at ranks 4, 8, 16       (~20 min)
make quantize              # merge + NF4 each rank         (~5 min)
LIMIT=200 make eval        # fast benchmark pass           (~15 min)
make analyze               # tables + figures
```

`make pipeline` chains all of it. Drop `LIMIT` for the full suite. Every stage is also a
standalone entry point, so a failed run resumes at the stage that failed rather than the top:

```bash
python -m efficient_slm.training.trainer --lora_rank 8
python -m efficient_slm.quantization.merge --lora_rank 8
python -m efficient_slm.quantization.quantize --model outputs/merged_r8 --output outputs/quantized_r8
python -m efficient_slm.inference.profiler
python -m efficient_slm.evaluation.benchmark --model outputs/quantized_r8 --name quantized_r8
```

The driver scripts are content-aware: `train_all_configs.sh` skips a rank whose adapter already
exists, `quantize_all.sh` skips a rank already merged or quantized, and `eval_all_checkpoints.sh`
skips a checkpoint that already has `scores.json`. Re-running after an interruption costs
nothing for work already done.

One trap in that: a `LIMIT=` pass writes a real `scores.json`, so the *next* full run skips
those checkpoints and you get a leaderboard built from 200-example samples. Clear
`eval/results/*/scores.json` (or `make clean`) between a smoke pass and the real one. The
`eval_limit` field recorded in every `scores.json` says which kind of run produced it.

## Pipeline

| Stage | What happens | Key tool |
|---|---|---|
| 0 | Preprocess `oasst1` into prompt/response pairs, alpaca template | `datasets` |
| 1 | QLoRA SFT — NF4 4-bit base, LoRA on the attention projections, one run per rank | `peft` + `transformers` Trainer |
| 2 | Merge each adapter into an fp16 checkpoint | `peft` `merge_and_unload` |
| 3 | 4-bit NF4 quantization of each merged checkpoint | `bitsandbytes` |
| 4 | Profile VRAM / latency / throughput on all 7 checkpoints | `torch.cuda` counters |
| 5 | Benchmark all 7 checkpoints on the identical suite | `lm-evaluation-harness` |
| 6 | Master table, deltas, efficiency metrics, figures | `pandas` / `matplotlib` |

**Why profiling is a separate stage from evaluation.** They answer different questions and fail
differently. Profiling is cheap (~1 min/checkpoint) and its numbers — VRAM, latency, model size —
are independent of whether the fine-tune learned anything. Evaluation is ~45 min/checkpoint and
only means something if training worked. Running the cheap one first turns "did the pipeline
produce loadable, correctly-sized checkpoints?" into a two-minute question instead of an
afternoon one.

**Why the SFT column is a merged fp16 model, not base+adapter.** The quantization arm has to
consume a single merged checkpoint anyway, so merging early means the SFT and quantized columns
differ by exactly one variable — the quantization — with no adapter-loading path in between to
account for.

## Benchmarks

| Task | Shots | Metric key | Measures |
|---|---|---|---|
| MMLU | 5 | `acc,none` | broad knowledge (57-subtask group) |
| ARC-Challenge | 25 | `acc_norm,none` | reasoning |
| HellaSwag | 10 | `acc_norm,none` | commonsense |
| GSM8K | 5 | `exact_match,flexible-extract` | math / multi-step |
| TruthfulQA MC2 | 0 | `acc,none` | truthfulness |
| **IFEval** | 0 | `inst_level_strict_acc,none` | **instruction-following** |

Shot counts follow the HF Open LLM Leaderboard convention, so the absolute numbers are
comparable to published results and not merely internally consistent.

**IFEval is load-bearing for this study.** The training data (`oasst1`) is general
instruction-following, and the other five benchmarks are knowledge and reasoning tasks that
instruction tuning barely moves. Without IFEval the expected result is five flat columns, and
"SFT did nothing" would be indistinguishable from "SFT worked but nothing here can see it."
Since fine-tuning also cannot move VRAM, latency or model size — a merged LoRA has the same
architecture and parameter count as its base — IFEval is close to the only channel in the whole
study where the SFT arm can register at all.

The metric keys are not interchangeable and picking the wrong one is a silent failure rather
than an error. GSM8K is generative and has no `,none` variant at all; ARC and HellaSwag are
length-normalized, so reading plain `acc` off them understates a small model by several points
and would show up as a fake quantization penalty. `METRIC_KEY_MAP` in
[`benchmark.py`](src/efficient_slm/evaluation/benchmark.py) pins one key per task.

MMLU is a *group* of 57 subtasks, not a task — lm-eval aggregates it under a single `mmlu` key,
and `--limit` applies per subtask, so `--limit 5` is 285 examples, not 5.

## Results

The efficiency axis is complete and reported below. The quality axis is **still running** — the
benchmark table is empty on purpose rather than filled from a partial or under-sampled run.

### Concluded

**1. Quantization cost — RQ3's cost side, and RQ4.** Measured across all seven checkpoints,
profiled in a single process with the peak-memory counter reset before each load:

| | VRAM | Latency | Throughput | On disk |
|---|---|---|---|---|
| Base / SFT (fp16) | 3.10–3.12 GB | 6.6 ms/token | ~129 tok/s | 3.10 GB |
| Quantized (NF4) | **1.57 GB** | 8.7 ms/token | ~107 tok/s | **1.16 GB** |

NF4 buys **−49% VRAM and −63% on disk for +32% latency**, reproduced identically at all three
ranks. Fine-tuning moves none of these figures, which is the correct result rather than a null
one: a merged LoRA has the same architecture and parameter count as its base, so SFT cannot
change the inference footprint. That is precisely why the quality axis is the only place the
SFT arm can register at all.

**2. LoRA rank barely affects training loss.** All three ranks trained identically — 4,000
examples, 250 optimizer steps, same hyperparameters, same seed:

| Rank | Trainable params | % of model | Final loss | Wall | Peak VRAM |
|---|---|---|---|---|---|
| r4 | 1,089,536 | 0.0705% | 1.4659 | 387 s | 4.21 GB |
| r8 | 2,179,072 | 0.1410% | 1.4628 | 379 s | 4.23 GB |
| r16 | 4,358,144 | 0.2815% | **1.4602** | 386 s | 4.26 GB |

Loss is monotonic in rank, but the spread across a **4× range of adapter capacity is 0.0057**,
and the trajectories run near-parallel from step 25 onward. On this dataset at one epoch,
capacity is not the binding constraint. This is a preliminary answer to RQ1 and it makes
`quality_per_param_million` the column to watch: r4 reaches effectively the same loss on a
quarter of the parameters.

Training loss is not a proxy for benchmark quality — it says the adapters converged similarly,
not that anything transferred. The table below is what settles that.

### Pending

| Benchmark | Base | SFT r4 | SFT r8 | SFT r16 | Quant r4 | Quant r8 | Quant r16 |
|---|---|---|---|---|---|---|---|
| MMLU (5-shot) | — | — | — | — | — | — | — |
| ARC-Challenge (25-shot) | — | — | — | — | — | — | — |
| HellaSwag (10-shot) | — | — | — | — | — | — | — |
| GSM8K (5-shot) | — | — | — | — | — | — | — |
| TruthfulQA MC2 (0-shot) | — | — | — | — | — | — | — |
| IFEval (0-shot) | — | — | — | — | — | — | — |

**Δ SFT and Δ Quant both have zero data points so far**, so the headline question — what SFT and
4-bit quantization actually do to the model — is unanswered. `make analyze` writes this table
plus `reports/results_table.csv` and `reports/delta_table.csv`; raw scores land in
`eval/results/<checkpoint>/scores.json`.

**IFEval is the row to watch.** It is the only benchmark here that instruction tuning should
meaningfully move. If Δ SFT is flat on IFEval too, that is a substantive negative result — general
instruction tuning on 4,000 `oasst1` pairs does not measurably improve instruction-following in a
1.5B model — rather than a missing one.

**Read the two deltas differently.**

- **Δ SFT** is fp16 vs fp16 — base and merged checkpoint, same loader, same precision. The
  adapter is the only variable. Clean.
- **Δ Quant** is fp16 vs NF4 — a genuine full-precision → 4-bit comparison, which this project
  *can* make because a 1.5B model fits in fp16 on a 16 GB card with room to spare. Larger-model
  studies usually can't and end up comparing two 4-bit schemes instead.

`oasst1` is general instruction-following, not benchmark-targeted, so the expected shape is
movement in GSM8K and roughly flat MMLU/HellaSwag. A large drop on the knowledge benchmarks
would indicate catastrophic forgetting rather than tuning.

## Verified environment

Measured on this machine, not estimated:

| | |
|---|---|
| GPU | NVIDIA RTX 5080, 16.3 GB, **sm_120** (Blackwell), 14.7 GB usable |
| torch | 2.13.0+cu130 — `sm_120` present in `get_arch_list()`, fp16 + bf16 confirmed on device |
| CUDA toolkit | driver only, **no `nvcc`** — nothing that needs source-compiled kernels can be used |
| Quantization | bitsandbytes 0.50.1 NF4 |
| Base model | Qwen2.5-1.5B-Instruct — 1.5437B params, 28 layers, **151,936-token vocab** |

**Measured pipeline cost:**

| Stage | Measured |
|---|---|
| QLoRA training | 1.54 s/optimizer-step, peak **4.23 GB** → **~7 min per rank** |
| Merge adapter | ~1 min |
| NF4 quantize | 3.10 GB → **1.16 GB (−63%)**, ~1 min |
| ARC-Challenge, full 5-shot | 4,687 loglikelihood requests in **53 s** (88 req/s) |
| GSM8K, 8-shot generative | **0.83 s/example** → ~18 min for all 1,319 |
| Full suite, one checkpoint | ~45 min at 4 benchmarks; re-measuring for the 6-benchmark suite |
| Full study, 7 checkpoints | ~5 h at 4 benchmarks; expect longer with IFEval + 25-shot ARC |

**Measured inference profile** (all seven in one process, after the reset fix below):

| Checkpoint | VRAM | Latency | Throughput | On disk |
|---|---|---|---|---|
| Base (fp16) | 3.10 GB | 6.6 ms/token | 129.0 tok/s | 3.10 GB |
| SFT r4 / r8 merged (fp16) | 3.12 GB | 6.6 ms/token | 129.6 / 130.1 tok/s | 3.10 GB |
| Quantized r4 / r8 (NF4) | **1.57 GB** | 8.6 / 8.7 ms/token | 108.0 / 107.1 tok/s | **1.16 GB** |

NF4 roughly halves resident VRAM for a ~16% latency cost. Both ranks land on identical
footprints, which is the expected result — rank changes trainable parameters, not inference
cost — and serves as a sanity check that the profiler is measuring the model rather than
leftover state. These figures are independent of training quality, so they stand; but the
adapters they came from are **short smoke runs**, not trained models. `make clean` clears them.

## Design decisions

**Model — Qwen2.5-1.5B-Instruct.** Small enough that all seven checkpoints fit the compute
budget, large enough to score meaningfully above chance on all four benchmarks. Critically, it
fits in **fp16**, which is what makes Δ Quant a real full-precision baseline rather than a
4-bit-vs-4-bit comparison. Its 151,936-token vocabulary is also the single biggest driver of
training memory here — see the batch-size note below.

**Dataset — `OpenAssistant/oasst1`, 5,000 pairs.** Human-written rather than model-distilled,
so the instruction-following signal isn't confounded with "learned to imitate GPT-4." The
loader reconstructs prompter→assistant pairs through `parent_id` and keeps **only the
top-ranked reply per prompt**, which matters: oasst1 stores multiple ranked replies per
prompt, and taking all of them floods the set with near-duplicate instructions and inflates
apparent data volume. 4,000 train / 1,000 val, seeded at 42, fully deterministic — re-running
`make data` reproduces the committed `data/stats.json` exactly.

**Adapters — ranks 4 / 8 / 16, alpha = 2×rank, dropout 0.05**, on `q_proj`, `k_proj`, `v_proj`,
`o_proj`. Measured trainable parameters:

| Rank | Trainable | % of model |
|---|---|---|
| 4 | 1,089,536 | 0.0705% |
| 8 | 2,179,072 | 0.1410% |
| 16 | 4,358,144 *(linear; r4 and r8 measured exactly 2× apart)* | 0.28% |

**Quantization — bitsandbytes NF4, not GPTQ.** This is a forced substitution, not a preference.
`auto-gptq` has published nothing since 0.7.1 (2024), ships **no cp312 wheel**, and its sdist
compiles CUDA kernels that predate Blackwell — and with no `nvcc` on the machine there is
nothing to build them with either. Three independent blockers. NF4 needs no compilation and
reuses the library QLoRA training already depends on, so the training and deployment arms share
a quantizer. The GPTQ path survives in
[`quantization/gptq.py`](src/efficient_slm/quantization/gptq.py) behind a lazy import; set
`method: "gptq"` in [`configs/quantize.yaml`](configs/quantize.yaml) on a machine where it
installs. `GPTQModel` is the maintained successor if you want GPTQ specifically, but it is
sdist-only and a source build without `nvcc` is a gamble.

**Optimizer — `paged_adamw_32bit`.** Verified working on sm_120; the paged variant is cheap
insurance against allocator spikes at the step boundary.

## Engineering notes

The interesting part of this project is what fitting a 16 GB card actually required. Each of
these is a real failure found by running the thing, not a hypothetical.

**The specified batch size never fit — on any targeted hardware.** The original config asked for
batch 8 × seq 2048 = 16,384 tokens per micro-batch. Qwen2.5's vocabulary is 151,936 tokens, so
the loss logits alone are `8 × 2048 × 151936 × 4 B` = **9.96 GB in fp32**, before weights,
activations or gradients. Measured ceiling, in isolated processes with gradient checkpointing on:

| Config | Tokens/micro-batch | Peak |
|---|---|---|
| 1 × 512 | 512 | 5.73 GB |
| 2 × 512 | 1,024 | 9.87 GB |
| 1 × 1024 | 1,024 | 10.58 GB |
| 2 × 1024 | 2,048 | **OOM** |

The real constraint is ~1024 tokens per micro-batch, not a batch size. The config is now
`batch 1 × 1024` with `grad_accum 16`, preserving the original effective batch of 16. Note the
plan targeted a T4 — **also 16 GB** — so this was never runnable as written, on Colab either.

**Sequence length is 1024 because of a measurement, not a guess.** Tokenized with the actual
Qwen tokenizer over all 4,000 training examples: mean 240, median 206, p90 442, p95 543, p99 899,
max 2368. A 1024 cap keeps **99.5% of examples intact and truncates 0.8% of all tokens** — the
memory is saved almost entirely from padding that was never carrying signal.

**`learning_rate: 2e-4` is a string.** YAML 1.1 requires the `2.0e-4` form; the bare version
parses as `str`. Nothing complains at config load. It surfaces minutes later, inside the
optimizer constructor, as:

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

Fixed in the YAML *and* coerced defensively in the trainer, because the failure mode is far
enough from the cause to cost a debugging session.

**Passing `quantization_config=None` destroys a quantized checkpoint's own config.** The
inference loader passed the kwarg unconditionally. For a checkpoint saved already-quantized —
i.e. half of what this project produces — the stored config gets overwritten with `None`, then:

```
AttributeError: 'NoneType' object has no attribute 'to_dict'
```

The loader now omits the kwarg entirely unless it is imposing a quantization config.

**The VRAM counter is peak-since-reset, and nothing was resetting it.** `max_memory_allocated()`
is monotonic within a process. Profiling seven checkpoints in one process meant every checkpoint
after the first reported the *previous* model's peak. Observed: `quantized_r8` reported
**3.87 GB** in a batch run and **1.57 GB** profiled alone. This is the most dangerous bug in the
set — it fails silently, produces plausible-looking numbers, and inflates precisely the
quantized checkpoints, which would have **inverted the RQ3 efficiency conclusion** while looking
entirely reasonable in the table. `profile_checkpoint` now resets peak stats before each load.

**The results table crashed on the normal stage order.** `build_results_dataframe` iterated only
checkpoints that had scores, so profiling-before-evaluation — the order the pipeline actually
runs in — produced an empty row list and `KeyError: "None of ['checkpoint'] are in the columns"`.
It now takes the union of scored and timed checkpoints.

**A test that could only pass without a GPU.** `test_measure_vram_returns_expected_shape` pins
its model to CPU, then asserted on `torch.cuda.is_available()` — a *machine* property — instead
of `model.device.type`. On a GPU-less box it passed; on this one it failed immediately. It was
itself evidence that nothing in the repo had ever been run against a GPU.

**Keras 3 aborts `transformers` on import.** This machine has TensorFlow installed for an
unrelated project, and `transformers` refuses to import when it finds Keras 3 without `tf-keras`.
`USE_TF=0` fixes it, but requiring every entry point to remember an env var is a trap, so
[`efficient_slm/__init__.py`](src/efficient_slm/__init__.py) sets it (along with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) at package import, before any submodule can
pull in `transformers`. `conftest.py` does the same for pytest.

**Isolated processes are the only honest way to measure OOM ceilings.** The first VRAM sweep
reported OOM for *every* config including ones that trivially fit. Cause: a caught
`OutOfMemoryError` keeps its traceback alive, the traceback keeps the frame, the frame keeps
every intermediate activation — so the first genuine OOM poisons the rest of the process, and
running the sweep back-to-back poisons the *next* process too because the driver hasn't
reclaimed yet. The numbers in the table above come from one process per config with a
free-memory settle check between them.

## Repository layout

```
├── configs/                  # model, data, LoRA, training, quantization, eval YAML
├── data/
│   ├── scripts/              # download + preprocess entry point
│   └── stats.json            # committed dataset statistics (regenerating reproduces it)
├── src/efficient_slm/
│   ├── data/loader.py        # oasst1 pair reconstruction, filtering, splitting
│   ├── training/             # QLoRA trainer + metrics callback
│   ├── quantization/         # merge, NF4 (bnb.py), GPTQ (lazy), dispatcher
│   ├── inference/            # load, generate, VRAM/latency/throughput profiling
│   ├── evaluation/           # lm-eval driver, metrics, comparison tables
│   └── utils/                # config loading, path resolution, GPU banner
├── scripts/                  # local pipeline drivers (CLI path)
├── notebooks/                # same pipeline, notebook path — 01-07, all local
├── eval/results/             # per-checkpoint scores.json + timing.json (gitignored)
├── reports/                  # results tables, writeup
├── figures/                  # generated plots
└── tests/                    # 18 unit tests, no GPU required
```

Every module under `src/` is importable *and* runnable as `python -m`. The `scripts/` drivers
are thin loops over those entry points — there is no logic in the shell layer, so anything you
can do with `make` you can do one stage at a time with the same code path.

## Notebooks

All seven notebooks run locally — no Colab, no Drive, no clone. Each is equivalent to a
`scripts/` driver and writes to the same paths, so the two interfaces are interchangeable:

| Notebook | Equivalent CLI |
|---|---|
| `01_data_exploration` | — (reads `data/processed/`) |
| `02_baseline_eval` | `python -m efficient_slm.evaluation.benchmark --model Qwen/Qwen2.5-1.5B-Instruct --name base` |
| `03_qlora_training` | `scripts/train_all_configs.sh` |
| `04_merge_quantize` | `scripts/quantize_all.sh` |
| `05_inference_profiling` | `python -m efficient_slm.inference.profiler` |
| `06_evaluation` | `scripts/eval_all_checkpoints.sh` |
| `07_analysis` | `scripts/analyze.py` |

Start Jupyter from the repo root (`jupyter lab`) so the relative paths resolve — each notebook
derives the repo root from its own location. `02` and `06` honour the same `LIMIT` environment
variable as the shell drivers.

The notebooks install nothing. Dependencies come from `make setup`, once, from the shell; the
first cell verifies versions and prints the GPU. That is deliberate — the original Colab cells
wrote `!pip install -q transformers>=4.44.0 ...`, where `>=` is a **shell redirect**, not a
version pin, so pip silently received bare package names and resolved to latest.

## Known divergences

Documented rather than fixed:

| Item | Status |
|---|---|
| `quantization/gptq.py` | Import-guarded. Cannot run on this machine; see the quantization decision above. |
| `configs/quantize.yaml` `gptq_config` | Retained for the GPTQ arm; ignored while `method: "bnb_nf4"`. |
| `configs/data.yaml` `max_length` | A *preprocessing* filter (2048), not the training memory budget. Training reads `max_seq_length` from `configs/train.yaml` (1024). |

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make test          # 18 tests, CPU only
```

`transformers` is pinned `<5` because peft and lm-eval have not all moved to the 5.x API.
`bitsandbytes>=0.45` is a hard floor, not a preference — it is the first release with sm_120
kernels, and older pins fail on any 50-series card. `auto-gptq` and `optimum` are deliberately
*not* dependencies; install them by hand only where they work.

Roughly 20 GB of disk is needed for the base model, three merged checkpoints, three quantized
checkpoints and the benchmark datasets.

## License

Code: MIT. Qwen2.5 weights are distributed under the Qwen license — review before
redistributing fine-tuned or quantized checkpoints.
