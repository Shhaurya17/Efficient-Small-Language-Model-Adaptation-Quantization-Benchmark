import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from efficient_slm.data.loader import format_pair
from efficient_slm.training.callbacks import MetricsCallback


def load_base_model_4bit(model_name, torch_dtype="float16"):
    dtype = getattr(torch, torch_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tokenizer


def build_lora_config(rank, alpha, dropout, target_modules, bias="none"):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="CAUSAL_LM",
    )


def tokenize_dataset(pairs, tokenizer, chat_template="alpaca", max_seq_length=2048):
    texts = [format_pair(p, chat_template) + tokenizer.eos_token for p in pairs]

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length)

    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def setup_qlora_trainer(model, train_dataset, eval_dataset, lora_config, train_config, tokenizer):
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        warmup_steps=train_config["warmup_steps"],
        logging_steps=train_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_config["eval_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        optim=train_config["optim"],
        max_grad_norm=train_config["max_grad_norm"],
        seed=train_config["seed"],
        fp16=train_config["mixed_precision"] == "fp16",
        gradient_checkpointing=train_config["gradient_checkpointing"],
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    metrics_callback = MetricsCallback()

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[metrics_callback],
    )
    trainer.metrics_callback = metrics_callback
    return trainer


def train(trainer, num_epochs=None):
    if num_epochs is not None:
        trainer.args.num_train_epochs = num_epochs
    return trainer.train()


def save_checkpoint(trainer, path):
    trainer.save_model(path)
    trainer.processing_class = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(path)
