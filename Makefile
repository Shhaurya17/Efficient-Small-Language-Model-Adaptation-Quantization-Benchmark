.PHONY: setup data train quantize eval analyze test clean

setup:
	pip install -r requirements.txt

data:
	python data/scripts/download_data.py

train:
	bash scripts/train_all_configs.sh

quantize:
	python src/efficient_slm/quantization/merge.py --config configs/lora.yaml
	python src/efficient_slm/quantization/gptq.py --config configs/quantize.yaml

eval:
	bash scripts/eval_all_checkpoints.sh

analyze:
	bash scripts/generate_report.sh

test:
	pytest tests/

clean:
	rm -rf outputs/* eval/results/**/*.json
