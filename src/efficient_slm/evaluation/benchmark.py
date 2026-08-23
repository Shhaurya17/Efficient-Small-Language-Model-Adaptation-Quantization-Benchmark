import json
import os

import lm_eval
from lm_eval.models.huggingface import HFLM

TASK_NAME_MAP = {"mmlu": "mmlu", "arc": "arc_challenge", "gsm8k": "gsm8k", "hellaswag": "hellaswag"}

# gsm8k has no ",none" metric key; hellaswag/arc use length-normalized acc
METRIC_KEY_MAP = {
    "mmlu": "acc,none",
    "arc": "acc_norm,none",
    "gsm8k": "exact_match,flexible-extract",
    "hellaswag": "acc_norm,none",
}


def run_lm_eval(model_path, benchmarks, dtype="float16", limit=None, device="cuda"):
    lm = HFLM(pretrained=model_path, dtype=dtype, device=device)
    scores, raw_results = {}, {}
    for bench in benchmarks:
        task = TASK_NAME_MAP[bench["name"]]
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=bench["num_fewshot"],
            batch_size=bench["batch_size"],
            limit=limit,
        )
        raw_results[bench["name"]] = result["results"][task]
        scores[bench["name"]] = result["results"][task][METRIC_KEY_MAP[bench["name"]]]
    return scores, raw_results


def parse_results(scores, raw_results, checkpoint_name, model_path, limit=None):
    return {
        "checkpoint": checkpoint_name,
        "model_path": model_path,
        "scores": scores,
        "raw_results": raw_results,
        "eval_limit": limit,
    }


def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
