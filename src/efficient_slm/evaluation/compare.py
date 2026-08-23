import pandas as pd

from efficient_slm.evaluation.metrics import calculate_delta_quant, calculate_delta_sft


def build_results_dataframe(scores_by_checkpoint, timing_by_checkpoint, training_metrics=None):
    training_metrics = training_metrics or {}
    rows = []
    for checkpoint, scores in scores_by_checkpoint.items():
        row = {"checkpoint": checkpoint}
        row.update(scores)

        timing = timing_by_checkpoint.get(checkpoint) or {}
        row["vram_gb"] = (timing.get("vram") or {}).get("peak_vram_gb")
        row["latency_ms_per_token"] = (timing.get("latency") or {}).get("ms_per_token")
        row["throughput_tokens_per_sec"] = (timing.get("throughput") or {}).get("throughput_tokens_per_sec")

        train_info = training_metrics.get(checkpoint) or {}
        row["trainable_params"] = train_info.get("trainable_params")

        rows.append(row)
    return pd.DataFrame(rows).set_index("checkpoint")


def compute_delta_table(df, benchmark_cols, sft_ranks=(4, 8, 16)):
    rows = []
    for rank in sft_ranks:
        sft_row, quant_row = f"sft_r{rank}", f"quantized_r{rank}"
        if sft_row not in df.index or quant_row not in df.index or "base" not in df.index:
            continue
        for bench in benchmark_cols:
            rows.append({
                "rank": rank,
                "benchmark": bench,
                "delta_sft": calculate_delta_sft(df.loc["base", bench], df.loc[sft_row, bench]),
                "delta_quant": calculate_delta_quant(df.loc[sft_row, bench], df.loc[quant_row, bench]),
            })
    return pd.DataFrame(rows)
