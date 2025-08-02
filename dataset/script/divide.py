import json, re, argparse, statistics

def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|\S", text, flags=re.UNICODE))

def detect_text_field(obj):
    for k in ["text", "function", "code", "content", "body"]:
        if k in obj and isinstance(obj[k], str):
            return k
    str_fields = [(k, len(v)) for k, v in obj.items() if isinstance(v, str)]
    if str_fields:
        return max(str_fields, key=lambda x: x[1])[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL path")
    ap.add_argument("--short_out", default="short.jsonl", help="output short JSONL")
    ap.add_argument("--long_out", default="long.jsonl", help="output long JSONL")
    args = ap.parse_args()

    # First pass: detect field and collect lengths
    detected_field = None
    lengths = []
    total = decode_errors = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if detected_field is None:
                detected_field = detect_text_field(obj) or "text"
            lengths.append(approx_token_count(obj.get(detected_field, "")))

    if not lengths:
        raise SystemExit("No valid samples parsed from input.")

    # Use median as threshold
    threshold = int(statistics.median(lengths))

    # Second pass: split
    short_cnt = long_cnt = 0
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.short_out, "w", encoding="utf-8") as f_s, \
         open(args.long_out, "w", encoding="utf-8") as f_l:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            L = approx_token_count(obj.get(detected_field, ""))
            if L <= threshold:
                f_s.write(json.dumps(obj, ensure_ascii=False) + "\n")
                short_cnt += 1
            else:
                f_l.write(json.dumps(obj, ensure_ascii=False) + "\n")
                long_cnt += 1

    # Report
    total_valid = short_cnt + long_cnt
    print("==== Split Done ====")
    print(f"Input: {args.input}")
    print(f"Detected text field: {detected_field}")
    print(f"Threshold (median approx tokens): {threshold}")
    print(f"Counts: short={short_cnt}, long={long_cnt}, long_ratio={(long_cnt/total_valid if total_valid else 0):.2%}")

if __name__ == "__main__":
    main()
