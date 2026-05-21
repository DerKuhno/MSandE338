import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1. Custom Dataset Class for Unlearning
class UnlearningDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []
        for text in texts:
            # Tokenize text with a fixed maximum length
            tokenized = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length", 
                return_tensors="pt"
            )
            # Flatten tensors from shape [1, seq_len] to [seq_len]
            self.examples.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            })
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]


