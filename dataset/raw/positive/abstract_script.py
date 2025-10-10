import json

def process_jsonl():
    """
    Process the jsonl file: starting from line 1, extract the first 10 lines from every 100-line block,
    process a total of 10,000 lines, and extract 1,000 lines in total
    """
    # Define input and output file paths
    input_file = 'positive_original.jsonl'
    output_file = 'positive.jsonl'
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Initialize counters
        line_count = 0
        extracted_count = 0
        
        # Read the first 10,000 lines
        while line_count < 10000:
            line = infile.readline()
            if not line:  # End of file
                break
            
            line_count += 1
            # Calculate which 100-line block the current line belongs to
            block_number = (line_count - 1) // 100
            # Calculate the position of the current line within its block
            position_in_block = (line_count - 1) % 100
            
            # If the line is among the first 10 lines in the block, write it to the output file
            if position_in_block < 10:
                outfile.write(line)
                extracted_count += 1
    
    print(f"Processing completed! A total of {line_count} lines were read, and {extracted_count} lines were extracted.")
    print(f"Extracted data has been saved to {output_file}")

if __name__ == "__main__":
    process_jsonl()    