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
def plot_relearning_curves(steps, vanilla_ppl, standard_unlearn_ppl, undo_ppl, cont_undo_ppl=None, refusal_undo_ppl=None, third_label="UNDO Framework (Strong/Traces Erased)"):
    plt.figure(figsize=(8, 5))

    # Plot curves
    plt.plot(steps, vanilla_ppl, label="Vanilla Pythia (Control Group)", color="gray", linestyle="--")
    plt.plot(steps, standard_unlearn_ppl, label="Standard Unlearning (Weak/Traces Left)", color="red")
    plt.plot(steps, undo_ppl, label=third_label, color="green", linewidth=2)
    if cont_undo_ppl is not None:
        plt.plot(steps, cont_undo_ppl, label="UNDO (continuous noise)", color="blue", linewidth=2)
    if refusal_undo_ppl is not None:
        plt.plot(steps, refusal_undo_ppl, label="UNDO (refusal training)", color="orange", linewidth=2)
    
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


def plot_retain_perplexity(labels, perplexities, filename="retain_perplexity.png",
                           title=None, ylabel=None, log_scale=False, ylim=None):
    """Bar chart comparing perplexity across model variants.

    Args:
        labels:       List of model names, e.g. ["Vanilla", "Unlearned", "UNDO"].
        perplexities: Corresponding list of perplexity values.
        filename:     Path to save the figure.
        title:        Chart title (auto-selected if None).
        ylabel:       Y-axis label (auto-selected if None).
        log_scale:    Use a log y-axis.  Useful when one bar is orders of
                      magnitude larger than the others (e.g. forget-set PPL
                      after gradient-ascent unlearning).
        ylim:         Optional (ymin, ymax) tuple to cap the y-axis.  Bars that
                      exceed ymax are drawn at ymax and annotated with their
                      true value + a "↑" marker so the reader knows they are
                      clipped.  Ignored when log_scale=True.
    """
    colors = ["#888888", "#e05a5a", "#e09a30", "#4c9e5c"][: len(labels)]
    fig, ax = plt.subplots(figsize=(8, 5))

    # When ylim is set, clamp bar heights so clipped bars don't vanish off-chart
    plot_ppls = list(perplexities)
    if ylim is not None and not log_scale:
        ymax = ylim[1]
        plot_ppls = [min(p, ymax) for p in perplexities]

    bars = ax.bar(labels, plot_ppls, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    if log_scale:
        ax.set_yscale("log")
    else:
        # Annotate each bar with its numeric value.
        # Clipped bars get the true value + ↑ to signal truncation.
        for bar, plot_ppl, true_ppl in zip(bars, plot_ppls, perplexities):
            clipped = ylim is not None and true_ppl > ylim[1]
            label_text = f"{true_ppl:.1f}↑" if clipped else f"{true_ppl:.1f}"
            y_pos = bar.get_height() - (bar.get_height() * 0.05) if clipped else bar.get_height() + 0.3
            va = "top" if clipped else "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                label_text,
                ha="center",
                va=va,
                fontsize=10,
                fontweight="bold",
                color="white" if clipped else "black",
            )

    if ylabel is None:
        ylabel = "Perplexity (log scale)" if log_scale else "Perplexity"
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Perplexity Comparison")

    if not log_scale:
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_ylim(0, max(perplexities) * 1.25)

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[Success] Chart saved as '{filename}'")
    plt.show()


