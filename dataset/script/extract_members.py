import json

def extract_members(input_path, member_output_path, non_member_output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        members = []
        non_members = []
        
        for line in f:
            if line.strip():  # Skip empty lines
                obj = json.loads(line)
                # Check the label to determine membership
                if obj.get('label') == 1:
                    members.append(obj)
                elif obj.get('label') == 0:
                    non_members.append(obj)

    # Write members to member.jsonl
    with open(member_output_path, 'w', encoding='utf-8') as f:
        for member in members:
            f.write(json.dumps(member, ensure_ascii=False) + "\n")

    # Write non-members to non-member.jsonl
    with open(non_member_output_path, 'w', encoding='utf-8') as f:
        for non_member in non_members:
            f.write(json.dumps(non_member, ensure_ascii=False) + "\n")

    print(f'Extracted {len(members)} members and {len(non_members)} non-members.')

if __name__ == "__main__":
    extract_members('dataset/python_sample.jsonl', 'dataset/member.jsonl', 'dataset/non-member.jsonl')