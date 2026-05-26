import torch
import math
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm



# 3. Core Function to Calculate Perplexity
def calculate_perplexity(model, tokenizer, texts, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize the input text string
            inputs = tokenizer(text, return_tensors="pt").to(device)
            labels = inputs["input_ids"].clone()
            
            # Forward pass through Pythia
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss # Sequence-averaged cross entropy loss
            
            num_tokens = inputs["input_ids"].size(1)
            
            # Accumulate total loss weighted by the number of tokens
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Calculate overall average cross-entropy loss
    avg_loss = total_loss / total_tokens
    # Perplexity is exponentiated cross-entropy loss
    perplexity = math.exp(avg_loss)
    return perplexity


# 5. Plotting Function for the Relearning Attack
# Use this function after tracking your loss during the relearning phase
def plot_relearning_curves(steps, vanilla_ppl, standard_unlearn_ppl, undo_ppl, cont_undo_ppl=None):
    plt.figure(figsize=(8, 5))

    # Plot curves
    plt.plot(steps, vanilla_ppl, label="Vanilla Pythia (Control Group)", color="gray", linestyle="--")
    plt.plot(steps, standard_unlearn_ppl, label="Standard Unlearning (Weak/Traces Left)", color="red")
    plt.plot(steps, undo_ppl, label="UNDO (one-shot noise)", color="green", linewidth=2)
    if cont_undo_ppl is not None:
        plt.plot(steps, cont_undo_ppl, label="UNDO (continuous noise)", color="blue", linewidth=2)
    
    # Formatting
    plt.xlabel("Relearning Gradient Steps")
    plt.ylabel("Forget Dataset Perplexity")
    plt.title("Relearning Attack Evaluation Curve")
    plt.yscale("log") # Perplexity scale is exponential; log scale makes it easy to read
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # Save chart
    plt.savefig("relearning_attack_results.png", dpi=300)
    print("\n[Success] Chart saved as 'relearning_attack_results.png'")
    plt.show()


