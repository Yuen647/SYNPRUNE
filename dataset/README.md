# Dataset

## Overview

The [**Python Function Benchmark**](https://huggingface.co/datasets/Sheerio/SynPrune-Python) serves as a real-world evaluation dataset for membership inference attacks on code LLMs, specifically targeting models pretrained on datasets like the Pile (e.g., Pythia, GPT-Neo, StableLM).  

The dataset contains training (member) data and non-training (non-member):  

- **Member data** includes 1,000 Python functions sampled from the Pile dataset (released in 2021). To ensure a diverse sample, we systematically selected **the first 10 functions** from every 100 consecutive entries in the Pile, resulting in a total of 1,000 member functions.

- **Non-member data** includes 1,000 Python functions extracted from 100 GitHub repositories created after January 1, 2024 (all four evaluated LLMs had been released prior to this date). To ensure repository quality, we sorted repositories by star count in descending order and extracted 10 Python functions from each repository in order.
  To verify that these functions were genuinely original and not cloned from pre-existing sources, we implemented a rigorous verification process: we parsed each candidate function's code using Python's `ast` module to extract its name, variable names, and function calls, then used these elements to build search queries for the GitHub API. The verification employed three heuristics: (1) searching for the exact function name to identify direct duplicates; (2) searching by internal variable names to detect refactored code reuse; and (3) searching for the complete string of function calls to find logic similarities. Two authors conducted peer reviews on the search results to ensure all 1,000 functions were original and created after January 2024.

The benchmark includes 214 non-member function files (some repositories contributed multiple files) with an average of 25.34 lines of code (LOC). For member functions, file counts are unavailable as this information was not provided in the Pile dataset.

The benchmark supports evaluation under varied member-to-non-member **ratios** (e.g., 1:1, 1:5, 5:1) and includes statistics on syntax conventions (e.g., **38.4%** of tokens are syntax-related across categories like data models and expressions).  

If you find this work helpful, please consider citing our paper:
```latex
@misc{li2025synprune,
    title={Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach},
    author={Yuanheng Li and Zhuoyang Chen and Xiaoyun Liu and Yuhao Wang and Mingwei Liu and Yang Shi and Kaifeng Huang and Shengjie Zhao},
    year={2025},
    eprint={2511.07033},
    archivePrefix={arXiv},
    primaryClass={cs.CR}
}
```


## divide.py

`divide.py` is a script designed to split a JSONL file into two separate files based on the approximate token count of a specified text field. It detects the appropriate text field from the input JSONL and uses the median token count as a threshold to categorize the entries into "short" and "long".

### Usage

To use `divide.py`, run the following command in your terminal:

```bash
python divide.py --input <input_jsonl_path> --short_out <output_short_jsonl_path> --long_out <output_long_jsonl_path>
```

- `--input`: Path to the input JSONL file (required).
- `--short_out`: Path to the output JSONL file for short entries (default: `short.jsonl`).
- `--long_out`: Path to the output JSONL file for long entries (default: `long.jsonl`).

## ratio.py

`ratio.py` is a script that creates datasets with specified positive and negative sample ratios from two JSONL files containing positive and negative samples. It randomly samples from the provided datasets to create a new dataset based on the defined configuration.

### Usage

To use `ratio.py`, simply run the script:

```bash
python ratio.py
```

This script will read from `positive/positive.jsonl` and `negative/negative.jsonl`, and create datasets based on the configurations defined in the script. The output files will be named `dataset_{name}.jsonl` for each configuration.

### Dataset Configurations

The following configurations are available in the script:

- `1_1`: 2000 total samples with a 1:1 positive to negative ratio.
- `1_5`: 1200 total samples with a 1:5 positive to negative ratio.
- `5_1`: 1200 total samples with a 5:1 positive to negative ratio.

## extract_members.py

`extract_members.py` is a script that extracts members and non-members from a JSONL file based on the `label` field. It reads from `python_sample.jsonl`, where a `label` of `1` indicates a member and a `label` of `0` indicates a non-member. The script outputs two separate JSONL files: one for members and one for non-members.

### Usage

To use `extract_members.py`, run the following command in your terminal:

```bash
python extract_members.py
```

This script will read from `dataset/python_sample.jsonl` and create the following output files:

- `dataset/member.jsonl`: Contains all entries with `label` equal to `1`.
- `dataset/non-member.jsonl`: Contains all entries with `label` equal to `0`.

### Output

After running the script, you will see a message indicating the number of extracted members and non-members.