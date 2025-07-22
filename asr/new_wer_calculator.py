import json
import jiwer
import os
import sys
import argparse
from normalizer.normalizer import EnglishTextNormalizer  # Updated import path

"""
Usage:
python new_wer_calculator.py \
    --ind_results /path/to/individual_results.json \
    --overall_results /path/to/overall_results.json \
    [--verbose]
"""




def calculate_wer(reference: str, hypothesis: str):
    """
    Compute Word Error Rate (WER) and its components between reference and hypothesis.

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
        '-ii', '--ind_results', required=True,
        help='Path to JSON file with individual results (one JSON entry per line)'
    )
    parser.add_argument(
        '-io', '--overall_results', required=True,
        help='Path to JSON file with overall results'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show detailed processing information'
    )
    args = parser.parse_args()

    # Validate input files
    for path in (args.ind_results, args.overall_results):
        if not os.path.isfile(path):
            sys.exit(f"Error: File not found: {path}")

    if args.verbose:
        print(f"Processing files: {args.ind_results} & {args.overall_results}")

    entries = []
    totals = {
        'wer_raw': 0.0, 'subs_raw': 0.0, 'del_raw': 0.0, 'ins_raw': 0.0, 'hits_raw': 0.0,
        'wer_norm': 0.0, 'subs_norm': 0.0, 'del_norm': 0.0, 'ins_norm': 0.0, 'hits_norm': 0.0,
        'ref_words': 0, 'pred_words': 0, 'norm_words': 0
    }
    count = 0

    # Read and process individual entries
    with open(args.ind_results, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                sys.exit(f"Error: Invalid JSON at line {line_num}")

            ref = entry.get('text', '')
            hyp = entry.get('pred_text', '')

            # Raw WER (no normalization)
            wer_raw, subs_r, del_r, ins_r, hits_r = calculate_wer(ref, hyp)
            entry['wer_raw'] = {
                'wer': wer_raw, 'substitutions': subs_r,
                'deletions': del_r, 'insertions': ins_r, 'hits': hits_r
            }

            # Normalized WER
            hyp_norm = normalizer(hyp)
            entry['pred_text_norm'] = hyp_norm
            wer_n, subs_n, del_n, ins_n, hits_n = calculate_wer(ref, hyp_norm)
            entry['wer_normalized'] = {
                'wer': wer_n, 'substitutions': subs_n,
                'deletions': del_n, 'insertions': ins_n, 'hits': hits_n
            }

            # Word counts
            ref_count = len(ref.split())
            hyp_count = len(hyp.split())
            norm_count = len(hyp_norm.split())
            entry['ref_word_count'] = ref_count
            entry['pred_word_count'] = hyp_count
            entry['pred_norm_word_count'] = norm_count

            # Accumulate totals
            totals['wer_raw'] += wer_raw
            totals['subs_raw'] += subs_r
            totals['del_raw'] += del_r
            totals['ins_raw'] += ins_r
            totals['hits_raw'] += hits_r
            totals['wer_norm'] += wer_n
            totals['subs_norm'] += subs_n
            totals['del_norm'] += del_n
            totals['ins_norm'] += ins_n
            totals['hits_norm'] += hits_n
            totals['ref_words'] += ref_count
            totals['pred_words'] += hyp_count
            totals['norm_words'] += norm_count

            entries.append(entry)
            count += 1

            if args.verbose:
                print(f"Entry {line_num}:")
                print(f"  RAW WER: {wer_raw:.2f}% | S:{subs_r}, D:{del_r}, I:{ins_r}, Hits:{hits_r}")
                print(f"  NORM WER:{wer_n:.2f}% | S:{subs_n}, D:{del_n}, I:{ins_n}, Hits:{hits_n}")

    # Write updated individual results
    with open(args.ind_results, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    # Update overall results
    with open(args.overall_results, 'r') as f:
        overall = json.load(f)

    # Add word counts
    overall['total_ref_words'] = totals['ref_words']
    overall['total_pred_words'] = totals['pred_words']
    overall['total_pred_norm_words'] = totals['norm_words']

    # Compute means
    if count > 0:
        overall_stats_raw = {
            'mean_wer_raw': totals['wer_raw'] / count,
            'mean_subs_raw': totals['subs_raw'] / count,
            'mean_del_raw': totals['del_raw'] / count,
            'mean_ins_raw': totals['ins_raw'] / count,
            'mean_hits_raw': totals['hits_raw'] / count
        }
        overall_stats_norm = {
            'mean_wer_norm': totals['wer_norm'] / count,
            'mean_subs_norm': totals['subs_norm'] / count,
            'mean_del_norm': totals['del_norm'] / count,
            'mean_ins_norm': totals['ins_norm'] / count,
            'mean_hits_norm': totals['hits_norm'] / count
        }
    else:
        overall_stats_raw = {}
        overall_stats_norm = {}

    # Write back overall results
    with open(args.overall_results, 'w') as f:
        json.dump(overall, f)
        f.write('\n')
        json.dump(overall_stats_raw, f)
        f.write('\n')
        json.dump(overall_stats_norm, f)
        f.write('\n')

    # Summary
    if count > 0:
        print(f"\nProcessed entries: {count}")
        print(f"Average Raw WER: {overall_stats_raw['mean_wer_raw']:.2f}%")
        print(f"Mean Raw Substitutions: {overall_stats_raw['mean_subs_raw']:.2f}")
        print(f"Mean Raw Deletions: {overall_stats_raw['mean_del_raw']:.2f}")
        print(f"Mean Raw Insertions: {overall_stats_raw['mean_ins_raw']:.2f}")
        print(f"Mean Raw Hits: {overall_stats_raw['mean_hits_raw']:.2f}")
        print(f"\nAverage Normalized WER: {overall_stats_norm['mean_wer_norm']:.2f}%")
        print(f"Mean Normalized Substitutions: {overall_stats_norm['mean_subs_norm']:.2f}")
        print(f"Mean Normalized Deletions: {overall_stats_norm['mean_del_norm']:.2f}")
        print(f"Mean Normalized Insertions: {overall_stats_norm['mean_ins_norm']:.2f}")
        print(f"Mean Normalized Hits: {overall_stats_norm['mean_hits_norm']:.2f}")
        print(f"\nUpdated files: {args.ind_results} & {args.overall_results}")
    else:
        print("No valid entries processed.")


if __name__ == '__main__':
    main()
