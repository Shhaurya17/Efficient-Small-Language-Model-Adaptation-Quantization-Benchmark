import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from efficient_slm.data.loader import prepare_data

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    stats = prepare_data(
        size=config["dataset_size"],
        output_path=config["processed_dir"],
        split_ratio=config["split_ratio"]["train"],
        min_length=config["preprocessing"]["min_length"],
        max_length=config["preprocessing"]["max_length"],
        chat_template=config["chat_template"],
    )

    print(f"Saved {stats['train_pairs']} train / {stats['val_pairs']} val pairs")
    print(f"Avg tokens: {stats['avg_tokens']:.1f}, max: {stats['max_tokens']}")


if __name__ == "__main__":
    main()
