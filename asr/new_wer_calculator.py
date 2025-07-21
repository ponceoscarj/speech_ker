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
        '-ii', '--ind_results',
        required=True,
        help='Path to json file with individual results'
    )

    parser.add_argument(
        '-io', '--overall_results',
        required=True,
        help='Path to json file with overall results'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed processing information'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.ind_results):
        sys.exit(f"Error: Input file not found: {args.ind_results}")
    
    if not os.path.isfile(args.overall_results):
        sys.exit(f"Error: Input file not found: {args.overall_results}")
    
    
    if args.verbose:
        print(f"Processing files: {args.ind_results} & {args.overall_results}")
    
    entries = []
    
    total_wer_raw = 0.0
    total_subs_raw = 0.0
    total_del_raw = 0.0
    total_ins_raw = 0.0
    total_hits_raw = 0.0

    total_wer_norm = 0.0
    total_subs_norm = 0.0
    total_del_norm = 0.0
    total_ins_norm = 0.0
    total_hits_norm = 0.0

    total_ref_words = 0.0
    total_pred_words = 0.0
    total_pred_norm_words = 0.0
    count = 0

    # Read and process entries
    with open(args.ind_results, 'r') as f:
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

                # Normalized WER
                hyp_normalized = normalizer(hyp)
                entry['pred_text_norm'] = hyp_normalized
                wer_norm = calculate_wer(ref, hyp_normalized)
                entry['wer_normalized'] = {
                    "wer": wer_norm[0],
                    "substitutions": wer_norm[1],
                    "deletions": wer_norm[2],
                    "insertions": wer_norm[3],
                    "hits": wer_norm[4]
                }

                # Add word counts
                ref_words = len(ref.split())
                hyp_words = len(hyp.split)
                entry['ref_word_count'] = ref_words
                entry['pred_word_count'] = hyp_words
                entry['pred_norm_word_count'] = hyp_normalized

                # Totals
                total_wer_raw += wer_raw[0]
                total_subs_raw += wer_raw[1]
                total_del_raw += wer_raw[2]
                total_ins_raw += wer_raw[3]
                total_hits_raw += wer_raw[4]
            
                total_wer_norm += wer_norm[0]
                total_subs_norm += wer_norm[1]
                total_del_norm += wer_norm[2]
                total_ins_norm += wer_norm[3]
                total_hits_norm += wer_norm[4]


                total_ref_words += ref_words
                total_pred_words += hyp_words
                total_pred_norm_words += hyp_normalized

                count += 1
                entries.append(entry)
                                
                if args.verbose:
                    print(f"Entry {line_num}:")
                    print(f"  RAW WER:         {wer_raw[0]:.2f}% | S:{wer_raw[1]}, D:{wer_raw[2]}, I:{wer_raw[3]}, Hits:{wer_raw[4]}")
                    print(f"  Normalized WER:  {wer_norm[0]:.2f}% | S:{wer_norm[1]}, D:{wer_norm[2]}, I:{wer_norm[3]}, Hits:{wer_norm[4]}")
                    
            except json.JSONDecodeError:
                sys.exit(f"Error: Invalid JSON at line {line_num}")

    # print('\nnew_wer_calculator.py\n',args.input_file, '\n')

    # Write back modified entries
    with open(args.ind_results, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    with open(args.overall_results, 'r') as f:
        overall = json.load(f)
        overall["mean_words_ref"] = total_ref_words
        overall["mean_words_pred"] = total_pred_words
        overall["mean_words_pred_norm"] = total_pred_norm_words

    overall_wer_raw = {
        "mean_wer_raw":   total_wer_raw/count,
        "mean_subs_raw":  total_subs_raw/count,
        "mean_del_raw":   total_del_raw/count,
        "mean_ins_raw":   total_ins_raw/count,
        "mean_hits_raw":  total_hits_raw/count
    }

    overall_wer_norm = {
        "wer_norm":  total_wer_norm/count,
        "subs_norm": total_subs_norm/count,
        "del_norm":  total_del_norm/count,
        "ins_norm":  total_ins_norm/count,
        "hits_norm": total_hits_norm/count
    }

    with open(args.overall_results, 'w') as f: 
        f.write(json.dumps(overall) + '\n')
        f.write(json.dumps(overall_wer_raw) + '\n')
        f.write(json.dumps(overall_wer_norm) + '\n')
    
    if count > 0:
        print(f"\nResults:")
        print(f"Processed entries: {count}")
        print(f"Average Raw WER: {overall_wer_raw['mean_wer_raw']:.2f}%")
        print(f"Average Normalized WER: {overall_wer_norm['wer_norm']:.2f}%")
        print(f"Updated files: {args.ind_results} & {args.overall_results}")
    else:
        print("No valid entries processed.")

if __name__ == "__main__":
    main()

