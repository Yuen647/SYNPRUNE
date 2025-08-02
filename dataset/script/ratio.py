import json
import random
import os

positive_path = 'positive/positive.jsonl'
negative_path = 'negative/negative.jsonl'

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line for line in f if line.strip()]

positives = read_jsonl(positive_path)
negatives = read_jsonl(negative_path)

datasets_config = {
    '1_1': {'total': 2000, 'pos_ratio': 1, 'neg_ratio': 1},  
    '1_5': {'total': 1200, 'pos_ratio': 1, 'neg_ratio': 5},  
    '5_1': {'total': 1200, 'pos_ratio': 5, 'neg_ratio': 1}   
}

for name, config in datasets_config.items():
    total = config['total']
    pos_ratio = config['pos_ratio']
    neg_ratio = config['neg_ratio']
    
    pos_count = int(total * pos_ratio / (pos_ratio + neg_ratio))
    neg_count = total - pos_count
    
    pos_count = min(pos_count, len(positives))
    neg_count = min(neg_count, len(negatives))
    
    pos_samples = random.sample(positives, pos_count)
    neg_samples = random.sample(negatives, neg_count)
    
    dataset = pos_samples + neg_samples
    random.shuffle(dataset)
    
    out_path = f'dataset_{name}.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line if line.endswith('\n') else line + '\n')
    
    print(f'{out_path} created: {pos_count} positive, {neg_count} negative samples (total: {len(dataset)})')
