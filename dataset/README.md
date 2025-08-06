# Dataset Scripts

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