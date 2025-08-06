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

get more info from ./dataset/README.md


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

get more info from ./src/README.md

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

get more info from ./src/README.md

## 🚀 How to Replicate

See the **./src/replicate.ipynb**

## evaluate

### ablate.py
This script runs **ablation experiments** for the SynPrune method.
It loads a Python function dataset and a language model, applies a **syntax-based masking strategy** where specific categories of syntax tokens (Data Model, Expressions, Single Statements, Compound Statements) can be selectively excluded via the `--drop` parameter, computes SynPrune scores, and outputs the corresponding performance metrics (AUROC, FPR\@95, TPR\@5).
It also saves results for each ablation setting to a CSV file for later analysis.

### visualization.py
This script loads a specified Python code dataset and a language model, computes **SynPrune** scores for each sample using a syntax-based token masking strategy, and outputs classification performance metrics such as AUROC and the best F1-score threshold.
It can also plot the **F1-score curve** across different thresholds, either saving it as an image file or displaying it directly, allowing intuitive analysis of the model’s performance under varying thresholds.

## 📌 Notes

- The script uses HuggingFace Transformers to load and run causal language models.
- You can replace `EleutherAI/pythia-2.8b` with any other HuggingFace CausalLM model (e.g., `gpt6j`, etc.).
- `synprune` is sensitive to syntax-based token masking. You can adapt the pruning rules in `get_closing_token_mask()`.