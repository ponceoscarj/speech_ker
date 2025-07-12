import json
import jiwer
import os
import sys
import argparse
from normalizer.normalizer import EnglishTextNormalizer  # Updated import path

'''
Usage
python new_wer_calculator.py --input-file /home/ext_ponceponte_oscar_mayo_edu/speech_ker/old_files/wer_mock_input.json
'''


def calculate_wer(reference, hypothesis):
    """
    Returns:
      wer_pct          – overall WER percentage
      n_substitutions – number of substitutions
      n_deletions     – number of deletions
      n_insertions    – number of insertions
      n_hits          – number of correct words
    """
    measures = jiwer.compute_measures(reference, hypothesis)
    wer_pct = measures["wer"] * 100
    return (
        wer_pct,
        measures["substitutions"],
        measures["deletions"],
        measures["insertions"],
        measures["hits"]
    )

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
    total_wer_norm = 0.0
    total_wer_raw = 0.0
    count = 0

    # Read and process entries
    with open(args.input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                ref = entry.get('text', '')
                hyp = entry.get('pred_text', '')

                # Raw WER (no normalization)
                wer_raw = calculate_wer(ref, hyp)
                entry['wer_raw'] = {
                    "wer": wer_raw[0],
                    "substitutions": wer_raw[1],
                    "deletions": wer_raw[2],
                    "insertions": wer_raw[3],
                    "hits": wer_raw[4]
                }
                total_wer_raw += wer_raw[0]

                # Normalized WER
                hyp_normalized = normalizer(hyp)
                entry['norm_pred_text'] = hyp_normalized
                wer_norm = calculate_wer(ref, hyp_normalized)
                entry['wer_normalized'] = {
                    "wer": wer_norm[0],
                    "substitutions": wer_norm[1],
                    "deletions": wer_norm[2],
                    "insertions": wer_norm[3],
                    "hits": wer_norm[4]
                }
                # Add word counts
                entry['ref_word_count'] = ref_wc
                entry['pred_word_count'] = pred_wc
                entry['norm_pred_word_count'] = norm_pred_wc

                # Totals
                total_wer_raw += wer_raw[0]
                total_wer_norm += wer_norm[0]
                total_ref_words += ref_wc
                total_pred_words += pred_wc
                total_norm_pred_words += norm_pred_wc

                count += 1
                entries.append(entry)
                                
                if args.verbose:
                    print(f"Entry {line_num}:")
                    print(f"  RAW WER:         {wer_raw[0]:.2f}% | S:{wer_raw[1]}, D:{wer_raw[2]}, I:{wer_raw[3]}, Hits:{wer_raw[4]}")
                    print(f"  Normalized WER:  {wer_norm[0]:.2f}% | S:{wer_norm[1]}, D:{wer_norm[2]}, I:{wer_norm[3]}, Hits:{wer_norm[4]}")
                    
            except json.JSONDecodeError:
                sys.exit(f"Error: Invalid JSON at line {line_num}")

    print('\nnew_wer_calculator.py\n',args.input_file, '\n')

    # Write back modified entries
    with open(args.input_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    if count > 0:
        print(f"\nResults:")
        print(f"Processed entries: {count}")
        print(f"Average Raw WER: {total_wer_raw / count:.2f}%")
        print(f"Average Normalized WER: {total_wer_norm / count:.2f}%")
        print(f"Updated file: {args.input_file}")
        print(f"New keys added: 'wer_raw', 'wer_normalized', 'norm_pred_text'")
    else:
        print("No valid entries processed.")

if __name__ == "__main__":
    main()
