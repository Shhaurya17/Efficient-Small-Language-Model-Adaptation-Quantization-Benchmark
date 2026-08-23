import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_and_adapter(base_model_name, adapter_path, torch_dtype="float16"):
    dtype = getattr(torch, torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    return peft_model, tokenizer


def merge_lora(peft_model):
    return peft_model.merge_and_unload()


def save_merged_model(merged_model, tokenizer, output_path):
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
