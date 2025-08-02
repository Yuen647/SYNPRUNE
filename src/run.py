import argparse
import ast
import json
import zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import auc, roc_curve

def extract_variable_names(code: str) -> dict:
    try:
        tree = ast.parse(code)
        variable_info = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    lines = code.split('\n')
                    if 0 <= node.lineno - 1 < len(lines):
                        line = lines[node.lineno - 1]
                        start_pos = node.col_offset
                        abs_start = sum(len(lines[i]) + 1 for i in range(node.lineno - 1)) + start_pos
                        abs_end = abs_start + len(var_name)
                        variable_info[var_name] = variable_info.get(var_name, []) + [(abs_start, abs_end)]
            elif isinstance(node, ast.FunctionDef):
                var_name = node.name
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    lines = code.split('\n')
                    if 0 <= node.lineno - 1 < len(lines):
                        line = lines[node.lineno - 1]
                        start_pos = line.find("def ", 0, node.col_offset) + 4
                        if start_pos >= 4:
                            abs_start = sum(len(lines[i]) + 1 for i in range(node.lineno - 1)) + start_pos
                            abs_end = abs_start + len(var_name)
                            variable_info[var_name] = variable_info.get(var_name, []) + [(abs_start, abs_end)]
            elif isinstance(node, ast.ClassDef):
                var_name = node.name
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    lines = code.split('\n')
                    if 0 <= node.lineno - 1 < len(lines):
                        line = lines[node.lineno - 1]
                        start_pos = line.find("class ", 0, node.col_offset) + 6
                        if start_pos >= 6:
                            abs_start = sum(len(lines[i]) + 1 for i in range(node.lineno - 1)) + start_pos
                            abs_end = abs_start + len(var_name)
                            variable_info[var_name] = variable_info.get(var_name, []) + [(abs_start, abs_end)]
        return variable_info
    except:
        return {}

def get_closing_token_mask(tokenizer, text: str, input_ids: torch.Tensor) -> torch.Tensor:
    CLOSE_KEYWORDS = {
        "else","elif","except","finally",
        "yield","yield from","break","continue",
        "pass","raise",
        "lambda","import","from","assert",
        "global","nonlocal",
        "in","is","as","and","or","not",
        "match","case",
        "if","for","while","try","with","class","def",
        "async","async for","async with","async def"
    }
    CLOSE_SYMBOLS = {")", "]", "}", ":", ",", ".", ";"}
    mask = torch.ones(input_ids.shape[1] - 1, dtype=torch.bool)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:])
    encoding = tokenizer.encode_plus(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        max_length=len(input_ids[0]),
        truncation=True,
        padding="max_length"
    )
    offset_mapping = encoding['offset_mapping'][1:len(tokens)+1]
    variable_positions = extract_variable_names(text)
    for i, token in enumerate(tokens):
        if token == '[PAD]':
            mask[i] = False
            continue
        if token.strip() != token:
            mask[i] = False
            continue
        if i < len(offset_mapping):
            token_start, token_end = offset_mapping[i]
        bare = token.strip()
        if bare in CLOSE_KEYWORDS or bare in CLOSE_SYMBOLS:
            mask[i] = False
        elif token.strip() == "_":
            mask[i] = False
        elif token.strip() == "." and i > 0 and i < len(tokens) - 1:
            prev_token = tokens[i-1].strip()
            if prev_token == "self" or any(prev_token == var_name for var_name in variable_positions):
                mask[i] = False
    return mask

def safe_compute_scores(token_log_probs: torch.Tensor, mask: torch.Tensor, ratio: float) -> float:
    valid_probs = token_log_probs[mask]
    if len(valid_probs) == 0:
        return 0.0
    valid_probs = valid_probs - valid_probs.max()
    valid_probs = torch.exp(valid_probs)
    valid_probs = valid_probs / valid_probs.sum()
    k = min(int(len(valid_probs) * ratio), len(valid_probs))
    return 0.0 if k == 0 else torch.mean(torch.sort(valid_probs)[0][:k]).item()

def load_jsonl_dataset(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def get_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    mask = ~np.isnan(scores)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
    auroc = auc(fpr, tpr)
    fpr95 = next((fpr[i] for i in range(len(tpr)) if tpr[i] >= 0.95), 1.0)
    tpr05 = next((tpr[i] for i in reversed(range(len(fpr))) if fpr[i] <= 0.05), 0.0)
    return auroc, fpr95, tpr05

def load_model(name: str, int8: bool, half: bool):
    int8_kwargs = {"load_in_8bit": True, "torch_dtype": torch.bfloat16} if int8 else {}
    half_kwargs = {"torch_dtype": torch.bfloat16} if half and not int8 else {}
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map="auto", **int8_kwargs, **half_kwargs
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = 1024
    if hasattr(tokenizer, 'do_lower_case'):
        tokenizer.do_lower_case = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--model", default="EleutherAI/pythia-2.8b")
    argp.add_argument("--dataset", default="python_sample.jsonl")
    argp.add_argument("--half", action="store_true")
    argp.add_argument("--int8", action="store_true")
    argp.add_argument("--max_length", type=int, default=512)
    args = argp.parse_args()
    model, tokenizer = load_model(args.model, args.int8, args.half)
    data = load_jsonl_dataset(args.dataset)
    scores_dict = defaultdict(list)
    for d in tqdm(data, desc="Processing"):
        raw = d["function"]
        encoding = tokenizer.encode_plus(
            raw,
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding='max_length',
            pad_to_multiple_of=8
        )
        inp = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        with torch.no_grad():
            out = model(inp, labels=inp, attention_mask=attention_mask)
            nll_raw = -out.loss.item()
        clen_raw = len(zlib.compress(raw.encode())) or 1
        scores_dict["loss_raw"].append(nll_raw)
        scores_dict["zlib_raw"].append(nll_raw / clen_raw)
        mask = get_closing_token_mask(tokenizer, raw, inp)
        with torch.no_grad():
            logits = model(inp, labels=inp, attention_mask=attention_mask).logits
            input_ids = inp[0][1:].unsqueeze(-1)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            masked_log_probs = log_probs.clone()
            masked_log_probs[~mask] = float('-inf')
            masked_log_probs = F.log_softmax(masked_log_probs, dim=-1)
            token_lp = masked_log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
            scores_dict["mink_1.0"].append(safe_compute_scores(token_lp, mask, 1.0))
    labels = [d.get("label", 0) for d in data]
    results = defaultdict(list)
    for k, v in scores_dict.items():
        auroc, fpr95, tpr05 = get_metrics(v, labels)
        results["method"].append(k)
        results["auroc"].append(f"{auroc:.1%}")
        results["fpr95"].append(f"{fpr95:.1%}")
        results["tpr05"].append(f"{tpr05:.1%}")
    df = pd.DataFrame(results)
    print("\nResults:\n", df)
    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{args.model.split('/')[-1]}.csv"
    df.to_csv(csv, mode="a" if csv.exists() else "w", index=False, header=not csv.exists())

if __name__ == "__main__":
    main()
