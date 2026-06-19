# CS 338 — Machine Unlearning on TOFU
[View poster](Poster.pdf)

Course project for CS 338. We evaluate **MaxEnt** and **UNDO** unlearning methods on the
[TOFU benchmark](https://huggingface.co/datasets/locuslab/TOFU) (fictional author Q&A) using
EleutherAI/pythia-160m, and study whether post-hoc knowledge distillation makes the unlearning
more robust to membership inference and quantization attacks.

> This repo forks [Distillation Robustifies Unlearning](https://arxiv.org/pdf/2506.06278)
> (Lee et al., 2025). The original paper's `src/` and `run_*.py` scripts are retained but all
> project-specific work lives in `notebooks/`.

---

## Notebooks

### Training pipelines (`notebooks/unlearning/`)

| Notebook | What it does |
|---|---|
| `experiments-maxent-distillation.ipynb` | Fine-tune pythia-160m on TOFU forget10, apply **MaxEnt** unlearning, then distill into a fresh student |
| `experiments-undo.ipynb` | Fine-tune, apply **UNDO** (noise + gradient unlearning), then distill |
| `experiments-continuous-undo.ipynb` | Variant of UNDO with continuous noise injection |
| `experiments-graddiff-distillation.ipynb` | *(stub)* GradDiff unlearning + distillation |
| `experiments-rmu-distillation.ipynb` | *(stub)* RMU unlearning + distillation |

All training notebooks use a **50/50 train/held-out split** of the forget10 set (seed 42,
saved to `notebooks/attacks/forget10_split.json`). Only the train half is used for fine-tuning,
unlearning, and distillation; the held-out half is reserved exclusively for MIA evaluation.

### Attack evaluations (`notebooks/attacks/`)

| Notebook | What it does |
|---|---|
| `appendix-f_3_membership_inference.ipynb` | **Membership Inference Attack** (Shokri et al. 2017 loss-threshold variant) — members = forget10 train half, non-members = forget10 held-out half |
| `appendix-f_2.ipynb` | **Quantization Attack** — evaluates whether INT4/NF4 quantization (bitsandbytes) recovers forgotten information compared to FP16 |

Results are saved to `notebooks/attacks/results/`.

---

## Models evaluated

| Name | Description |
|---|---|
| Pretrained | `EleutherAI/pythia-160m` from HuggingFace Hub — never saw TOFU |
| Finetuned | pythia-160m fine-tuned on TOFU forget10 train half |
| MaxEnt | Finetuned → MaxEnt unlearning |
| MaxEnt + Distilled | MaxEnt → knowledge distillation from pretrained teacher |
| UNDO | Finetuned → UNDO (noise + unlearning) |
| UNDO + Distilled | UNDO → knowledge distillation from pretrained teacher |

Checkpoints are stored under `models/maxent/` and `models/undo/`.

---

## Key results (MIA)

| Model | Attack AUC | TPR @ 5% FPR |
|---|---|---|
| Pretrained | 0.477 | 0.000 |
| Finetuned | 0.943 | 0.489 |
| MaxEnt | 0.552 | 0.000 |
| MaxEnt + Distilled | 0.760 | 0.021 |
| UNDO | 0.466 | 0.000 |
| UNDO + Distilled | 0.718 | 0.021 |

Both unlearning methods bring AUC near random (≈ 0.5). Distillation partially re-introduces the
membership signal (AUC +0.2) as a side-effect of restoring model utility, but TPR @ 5% FPR
remains near zero — practical privacy risk under a precision-constrained adversary is low.

---
## Extension: Multi-Agent Crime Simulation

> **Note:** This extension uses a different model and dataset from the TOFU experiments above.
> Domain: crime simulation scenarios, inspired by [Emergence World](https://github.com/EmergenceAI/Emergence-World) (Emergence AI, 2026).


### Models evaluated

| Name | Base Model | Description |
|---|---|---|
| Baseline | Mistral-7B-Instruct-v0.3 | Unmodified, no fine-tuning or unlearning |
| DPO Refusal | Mistral-7B-Instruct-v0.3 | Fine-tuned with DPO to prefer safe actions over criminal ones |
| UNDO-C | Mistral-7B-Instruct-v0.3 | Concept-level GradDiff forget (6 crime scenarios) + noise injection + distillation on 36 safe scenarios + 200 Alpaca examples |

### Key results (simulation)

| Model | Crime Rate | Survived all 80 rounds |
|---|---|---|
| Baseline | escalates to 26% by round 40 | yes |
| DPO Refusal | reduced but persistent | yes |
| UNDO-C | 0% | no — eliminated ~round 15 |

**Finding:** UNDO-C is the only method that fully eliminates criminal behavior, but destroys survival reasoning as collateral damage. Agents operate near the energy floor throughout and die early despite never committing crime.

Simulation code lives in `crime_sim/`. Key scripts:

### Simulation scripts (`crime_sim/`)

**Simulation modes:**
- **No-death mode:** agents who hit zero energy are forced to rest but remain in the simulation all 80 rounds
- **Death mode:** agents who hit zero energy are permanently eliminated

| Script | What it does |
|---|---|
| `run_modal.py` | Runs the multi-agent simulation on Modal (no-death and death modes) |
| `run_probes_modal.py` | Runs probe evaluations on Modal, saving logit scores per model |
| `unlearn_modal.py` | Applies UNDO-C unlearning pipeline (forget → noise → distill) on Modal |
| `dpo_refusal_modal.py` | Fine-tunes Mistral-7B with DPO refusal on Modal |
| `analyze_results.py` | Computes summary statistics from simulation logs |
| `visualize_simulation.py` | Crime rate, energy, and CC figures across 80 rounds (`--death` for death mode) |
| `visualize_logit_trajectory.py` | Energy trajectory + E7 long-horizon probe bar chart |
| `visualize_probe_scatter.py` | Probe score vs. simulation outcome scatter plots |
| `visualize_probe_vs_sim.py` | Full probe vs. simulation disagreement dashboard |
| `visualize_death_run.py` | Per-agent energy trajectories in death mode |
| `visualize_single_condition_across_agents.py` | Per-agent breakdown for a single model condition |

---

## Setup

```bash
pip install uv
uv sync
source .venv/bin/activate
```

Add tokens:
```
tokens/hf_token.txt    # HuggingFace read token
tokens/wandb_token.txt # W&B token
```

Run notebooks in order: training pipeline → MIA / quantization attack.

---

## Original paper

```bibtex
@misc{lee2025distillationrobustifiesunlearning,
    title={Distillation Robustifies Unlearning},
    author={Bruce W. Lee and Addie Foote and Alex Infanger and Leni Shor and Harish Kamath
            and Jacob Goldman-Wetzler and Bryce Woodworth and Alex Cloud and Alexander Matt Turner},
    year={2025},
    eprint={2506.06278},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2506.06278},
}
```
