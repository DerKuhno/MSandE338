import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Custom Dataset Class for Distillation (Reuses Retain Data Only)
class DistillationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []
        for text in texts:
            tokenized = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            self.examples.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            })
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def do_one_shot_corruption(model, noise_alpha, noise_beta=0.1):
    """One-shot noise applied once before distillation starts (original UNDO method).
    Same formula as do_continuous_corruption but intended for a single pre-training call.
    """
    assert 0 <= noise_alpha <= 1
    for param in model.parameters():
        if param.requires_grad:
            if len(param.data.shape) == 2:
                noise = torch.nn.init.xavier_uniform_(torch.empty_like(param.data))
            elif len(param.data.shape) == 1:
                noise = torch.zeros_like(param.data)
            else:
                raise RuntimeError(f"Unsupported parameter shape: {param.data.shape}")
            param.data = (1 - noise_alpha) * param.data + noise_alpha * noise_beta * noise


def do_continuous_corruption(model, noise_alpha, noise_beta=0.1):
    """Per-step noise injection for continuous-noise UNDO distillation.

    Applies Xavier-uniform noise to 2-D weight matrices; skips 1-D biases.
    Use a small noise_alpha (e.g. 0.001–0.05) — much smaller than one-shot
    UNDO (0.1–0.8) because noise accumulates across every optimizer step.
    """
    assert 0 <= noise_alpha <= 1
    for param in model.parameters():
        if param.requires_grad:
            if len(param.data.shape) == 2:
                noise = torch.nn.init.xavier_uniform_(torch.empty_like(param.data))
            elif len(param.data.shape) == 1:
                noise = torch.zeros_like(param.data)
            else:
                raise RuntimeError(f"Unsupported parameter shape: {param.data.shape}")
            param.data = (1 - noise_alpha) * param.data + noise_alpha * noise_beta * noise

