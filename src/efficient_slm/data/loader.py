import json
import random
from pathlib import Path

from datasets import load_dataset

ALPACA_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
QWEN_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
DEFAULT_TEMPLATE = "[INST] {instruction} [/INST] {response}"

TEMPLATES = {
    "alpaca": ALPACA_TEMPLATE,
    "qwen": QWEN_TEMPLATE,
    "default": DEFAULT_TEMPLATE,
}


def load_openassistant_subset(size=5000, seed=42):
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {row["message_id"]: row for row in dataset}

    pairs = []
    for row in dataset:
        if row["role"] != "assistant" or row["lang"] != "en":
            continue
        parent = by_id.get(row["parent_id"])
        if parent is None or parent["role"] != "prompter" or parent["lang"] != "en":
            continue
        pairs.append({
            "instruction": parent["text"].strip(),
            "response": row["text"].strip(),
            "rank": row.get("rank") if row.get("rank") is not None else 999,
        })

    # Keep only the top-ranked (best) reply per prompt to avoid near-duplicate pairs
    best_by_instruction = {}
    for pair in pairs:
        key = pair["instruction"]
        if key not in best_by_instruction or pair["rank"] < best_by_instruction[key]["rank"]:
            best_by_instruction[key] = pair

    pairs = [{"instruction": p["instruction"], "response": p["response"]} for p in best_by_instruction.values()]

    random.Random(seed).shuffle(pairs)
    return pairs[:size]


def filter_valid_pairs(pairs, min_length=10, max_length=2048):
    valid = []
    for pair in pairs:
        instruction, response = pair["instruction"], pair["response"]
        if not instruction or not response:
            continue
        response_len = len(response.split())
        if response_len < min_length or response_len > max_length:
            continue
        valid.append(pair)
    return valid


def remove_duplicates(pairs):
    seen = set()
    deduped = []
    for pair in pairs:
        key = (pair["instruction"], pair["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    return deduped


def train_val_split(pairs, ratio=0.8, seed=42):
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_to_disk(pairs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def format_pair(pair, chat_template="alpaca"):
    template = TEMPLATES[chat_template]
    return template.format(instruction=pair["instruction"], response=pair["response"])


def compute_stats(train_pairs, val_pairs, chat_template="alpaca"):
    all_pairs = train_pairs + val_pairs
    token_counts = [len(format_pair(p, chat_template).split()) for p in all_pairs]
    instruction_lengths = [len(p["instruction"].split()) for p in all_pairs]
    response_lengths = [len(p["response"].split()) for p in all_pairs]
    return {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths),
        "avg_response_length": sum(response_lengths) / len(response_lengths),
    }


def prepare_data(
    size=5000,
    output_path="data/processed",
    split_ratio=0.8,
    min_length=10,
    max_length=2048,
    chat_template="alpaca",
    seed=42,
):
    pairs = load_openassistant_subset(size=size * 2, seed=seed)  # oversample before filtering
    pairs = filter_valid_pairs(pairs, min_length=min_length, max_length=max_length)
    pairs = remove_duplicates(pairs)
    pairs = pairs[:size]

    train_pairs, val_pairs = train_val_split(pairs, ratio=split_ratio, seed=seed)

    output_dir = Path(output_path)
    save_to_disk(train_pairs, output_dir / "train.jsonl")
    save_to_disk(val_pairs, output_dir / "val.jsonl")

    stats = compute_stats(train_pairs, val_pairs, chat_template=chat_template)
    with open(output_dir.parent / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats
