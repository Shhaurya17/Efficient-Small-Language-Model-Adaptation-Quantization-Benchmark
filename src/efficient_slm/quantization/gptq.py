from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

CALIBRATION_DATASET_MAP = {"wikitext": "wikitext2"}


def prepare_calibration_dataset(dataset_name="wikitext", size=128):
    return CALIBRATION_DATASET_MAP.get(dataset_name, dataset_name)


def load_merged_model(model_path, torch_dtype="float16"):
    import torch

    dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def quantize_model_gptq(model, tokenizer, config):
    dataset = prepare_calibration_dataset(config.get("calibration_dataset", "wikitext"))
    quantizer = GPTQQuantizer(
        bits=config["gptq_config"]["bits"],
        group_size=config["gptq_config"]["group_size"],
        desc_act=config["gptq_config"]["desc_act"],
        dataset=dataset,
    )
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model, quantizer


def save_quantized_model(quantized_model, quantizer, tokenizer, output_path):
    quantizer.save(quantized_model, output_path)
    tokenizer.save_pretrained(output_path)


def verify_quantized_model(model_path, prompt="Hello, how are you?", max_new_tokens=20):
    model, tokenizer = load_merged_model(model_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
