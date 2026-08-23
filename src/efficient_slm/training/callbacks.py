import time

import torch
from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.training_time_sec = None
        self.peak_vram_gb = 0.0
        self.loss_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            self.peak_vram_gb = max(self.peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9)
        if logs and "loss" in logs:
            self.loss_history.append({"step": state.global_step, "loss": logs["loss"]})

    def on_train_end(self, args, state, control, **kwargs):
        self.training_time_sec = time.perf_counter() - self.start_time

    def summary(self):
        return {
            "training_time_sec": self.training_time_sec,
            "peak_vram_gb": self.peak_vram_gb if torch.cuda.is_available() else None,
            "loss_history": self.loss_history,
            "final_loss": self.loss_history[-1]["loss"] if self.loss_history else None,
        }
