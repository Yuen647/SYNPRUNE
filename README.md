# SYNPRUNE

This is the official repository for the paper Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach.

## Overview

We propose **SYNPRUNE**, a syntax-pruned membership inference attack method tailored for code, to detect whether specific code samples were included in the pretraining data of large language models (LLMs), addressing transparency, accountability, and copyright compliance issues in code LLMs. Unlike prior membership inference attack (MIA) methods that treat code as plain text, **SYNPRUNE** leverages the structured nature of programming languages by pruning consequent tokens dictated by syntax conventions (e.g., from Python's data models, expressions, statements), excluding them from attribution scores to improve detection accuracy. To evaluate pretraining data detection for code LLMs, we introduce a new benchmark of Python functions, sourced from the **Pile dataset** for members and post-2024 GitHub repositories for non-members.

![overview](./assets/overview.png)

## Benchmark

The **Python Function Benchmark** serves as a real-world evaluation dataset for membership inference attacks on code LLMs, specifically targeting models pretrained on datasets like the Pile (e.g., Pythia, GPT-Neo, StableLM).  

The dataset contains non-training (non-member) and training (member) data:  

- **Non-member data** includes 1000 Python functions extracted from 214 GitHub repositories created after January 1, 2024 (post-release cutoff for evaluated LLMs), verified for originality using heuristics like function name searches, variable name checks, and logic similarity detection via GitHub API.  
- **Member data** includes 1000 Python functions randomly sampled from the Pile dataset (released in 2021), which is widely used in LLM pretraining.  

The benchmark supports evaluation under varied member-to-non-member **ratios** (e.g., 1:1, 1:5, 5:1) and includes statistics on syntax conventions (e.g., **38.4%** of tokens are syntax-related across categories like data models and expressions).  



## 🔧 Environment Setup

First, install dependencies:

```bash
pip install -r requirements.txt
```

You are recommended to use **Python 3.10+** and a machine with GPU support for large language models like Pythia.

------

## 📂 Dataset Format

Your input dataset should be in `.jsonl` format, where each line is a JSON object:

```json
{"function": "def foo():\n    return 1", "label": 1}
```

- `"function"`: the raw Python code string to be scored
- `"label"`: binary label (0 or 1) used for computing AUROC and related metrics

Default file name is `python_sample.jsonl`, or you may specify it using `--dataset`.

## 🚀 How to Run

The main evaluation script is `run.py`. You can execute it as follows:

```bash
python run.py \
  --model EleutherAI/pythia-2.8b \
  --dataset python_sample.jsonl
```

### Optional arguments:

- `--half`: enable bfloat16 inference (recommended if your GPU supports it)
- `--int8`: use 8-bit inference (requires `bitsandbytes`)
- `--max_length`: maximum sequence length (default is 512)

## 📊 Output

After running, a table will be printed to the console showing the evaluation metrics:

- **AUROC**: Area under the ROC curve
- **FPR@95**: False Positive Rate at 95% TPR
- **TPR@5**: True Positive Rate at 5% FPR

The script currently computes and logs scores for:

- `loss`: model negative log-likelihood
- `zlib`: compression-normalized log-likelihood
- `mink_0.2`: bottom-20% average log-probability (unmasked)
- `synprune`: masked average log-probability using a syntax-based pruning mask

## 🚀 How to Replicate

See the **replicate.ipynb**

## 📌 Notes

- The script uses HuggingFace Transformers to load and run causal language models.
- You can replace `EleutherAI/pythia-2.8b` with any other HuggingFace CausalLM model (e.g., `gpt6j`, etc.).
- `synprune` is sensitive to syntax-based token masking. You can adapt the pruning rules in `get_closing_token_mask()`.