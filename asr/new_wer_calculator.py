import json
import jiwer
import os
import sys
import argparse
from normalizer.normalizer import EnglishTextNormalizer  # Updated import path

'''
Usage
python new_wer_calculator.py --input-file /path/to/your/file.json
'''


def calculate_wer(reference, hypothesis):
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    
    if not reference:
        return 100.0 if hypothesis else 0.0
    try:
        return jiwer.wer(reference, hypothesis) * 100
    except ZeroDivisionError:
        return 100.0

def main():
    # Initialize text normalizer
    normalizer = EnglishTextNormalizer()

    parser = argparse.ArgumentParser(
        description='Calculate Word Error Rate (WER) for JSON entries with text normalization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input-file',
        required=True,
        help='Path to input JSON file (JSON lines format)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed processing information'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        sys.exit(f"Error: Input file not found: {args.input_file}")
    
    if args.verbose:
        print(f"Processing file: {args.input_file}")
    
    entries = []
    total_wer = 0.0
    count = 0

    # Read and process entries
    with open(args.input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                ref = entry.get('text', '')
                hyp = entry.get('pred_text', '')
                
                # Normalize hypothesis text
                hyp_normalized = normalizer(hyp)
                entry['norm_pred_text'] = hyp_normalized
                
                wer = calculate_wer(ref, hyp_normalized)
                entry['wer_normalized'] = wer
                
                entries.append(entry)
                total_wer += wer
                count += 1
                
                if args.verbose:
                    print(f"Processed entry {line_num}. Normalized WER={wer:.2f}%\n")
                    # print(f"Original hyp: {hyp}")
                    # print(f"Normalized hyp: {hyp_normalized}")
                    # print(f"WER: {wer:.2f}%\n")
                    
            except json.JSONDecodeError:
                sys.exit(f"Error: Invalid JSON at line {line_num}")

    # Write back modified entries
    with open(args.input_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    if count > 0:
        print(f"\nResults:")
        print(f"Processed entries: {count}")
        print(f"Average Normalized WER: {total_wer / count:.2f}%")
        print(f"Updated file: {args.input_file}")
        print(f"New keys added: 'wer_normalized', 'norm_pred_text'")        
    else:
        print("No valid entries processed.")

if __name__ == "__main__":
    main()
