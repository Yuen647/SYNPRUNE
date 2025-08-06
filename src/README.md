This script evaluates a Python code dataset using multiple scoring metrics derived from language model predictions.
It implements syntax-based masking (**SynPrune**) following the syntax conventions defined in the reference paper.

## Main steps

1. **Load** a causal language model and tokenizer from HuggingFace Transformers.
2. **Read** a dataset of Python functions in JSONL format.
3. **Compute** four metrics for each sample:

   * **Loss** (model log-likelihood)
   * **Zlib** (loss normalized by compressed length)
   * **Mink(0.2)** (Minkowski score on raw tokens)
   * **SynPrune** (Minkowski score after syntax-based token masking)
4. **Output** AUROC, FPR\@95, and TPR\@5 for each metric.
5. **Save** results to a CSV file.

---

## DC-PDD Reproduction Steps

To reproduce the original **DC-PDD** experiments, you can follow these steps:

```bash
# Clone the DC-PDD repository
git clone https://github.com/zhang-wei-chao/DC-PDD.git
cd ./DC-PDD/src/

# 1. Compute Program Discrepancy
python com_pro_dis.py \
  --tar_mod your_model \
  --ref_mod your_model \
  --data ../data/your_dataset \
  --gpu_num 1 \
  --key_nam function

# 2. Compute Frequency Discrepancy
python com_fre_dis.py \
  --data_file ../data/your_dataset \
  --field function \
  --model your_model \
  --max_tok 1024

# 3. Compute Detection Score
python com_det_sco.py \
  --tar_mod your_model \
  --data your_dataset \
  --max_cha 512 \
  --lang en \
  --a 0.01
```

---

## Adjustments in This Script Compared to DC-PDD

* **Metric change**:
  DC-PDD uses program discrepancy, frequency discrepancy, and detection score,
  while this script computes **Loss**, **Zlib**, **Mink(0.2)**, and **SynPrune**.
* **Syntax masking**:
  Implements **SynPrune** based on detailed syntax conventions from the paper, which is not present in DC-PDD.
* **Dataset**:
  Compared to raw DC-PDD, our version uses dataset `python_sample.jsonl`
