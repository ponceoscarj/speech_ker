import argparse
import json
import os

"""
Script to create manifests for any task. It can be modified as needed.

Difference between asr and dataset type is that dataset has fewer variables. 
For Canary models, you need to use the "asr" type. 

For ASR:
python create_manifest.py asr \
    --data_dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/audio_files/toy \
    --work_dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/nvidia-nemo \
    --output_file asr_manifest \
    --source_lang en \
    --target_lang en \
    --include_ground_truth \
    --pnc yes

For diarization:
python script.py diarization \
    --data_dir /path/to/audio_files \
    --work_dir /output/directory \
    --output_file diar_manifest \
    --url_transcript_text /path/to/transcript.txt

For dataset
python script.py dataset \
    --data_dir /path/to/audio_files \
    --work_dir /output/directory \
    --output_file dataset_manifest

For aligner
python script.py aligner \
    --data_dir /path/to/audio_files \
    --work_dir /output/directory \
    --output_file align_manifest \
    --url_transcript_text /path/to/transcript.txt        


Example for asr with ground truth
python create_manifest.py asr \
    --data_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
    --work_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/work_files \
    --output_file asr_manifest \
    --source_lang en \
    --target_lang en \
    --include_ground_truth \
    --pnc no    

python create_manifest.py diarization \
    --data_dir /home/ext_alzahidy_misk_mayo_edu/speech_ker/audio_files/audio_valid \
    --work_dir /home/ext_alzahidy_misk_mayo_edu/speech_ker/asr/work_files \
    --output_file diarize_manifest

"""


def create_manifest(data_dir, work_dir, output_file_name, 
                    manifest_type, url_transcript_text=None, 
                    source_lang="en", target_lang="en", pnc="no",
                    include_ground_truth=False):
    
    manifest_data = []

    wav_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.wav'))

    transcript_lines = []
    if url_transcript_text:
        if os.path.isdir(url_transcript_text):
            all_txt_files = sorted([os.path.join(url_transcript_text, f) for f in os.listdir(url_transcript_text) if f.endswith('.txt')])
            for txt_file in all_txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    transcript_lines.append(f.read().strip())
        elif os.path.isfile(url_transcript_text):
            with open(url_transcript_text, 'r', encoding='utf-8') as f:
                transcript_lines = [line.strip() for line in f.readlines()]

    for i, wav_file in enumerate(wav_files):
        base = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(data_dir, wav_file)
        txt_file = base + '.txt'
        txt_path = os.path.join(data_dir, txt_file)
        exists_txt = os.path.isfile(txt_path)
        
        entry = None

        if manifest_type == "asr":
            # Read ground-truth text from corresponding .txt file
            
            entry = {
                "audio_filepath": os.path.join(data_dir, wav_file),
                "duration": None,
                "taskname": "asr",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "pnc": pnc
            }

            if include_ground_truth:
                txt_file = base + ".txt"
                txt_path = os.path.join(data_dir, txt_file)

                try:
                    with open(txt_path, "r") as f:
                        ground_truth_text = f.read().strip()
                    entry["text"] = ground_truth_text
                except FileNotFoundError:
                    print(f"Warning: Missing ground truth file {txt_path}")
                    entry["text"] = "-"
            else:
                entry["text"] = "-"


        elif manifest_type == "diarization":
            txt = transcript_lines[i] if i < len(transcript_lines) else None
            entry = {
                "audio_filepath": wav_path,
                "duration": None,
                "offset": 0,
                "label": "infer",
                "text": txt,
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None
            }
        elif manifest_type == "aligner":
            # one TXT per WAV in the same directory:
            base, _ = os.path.splitext(wav_file)
            txt_path = os.path.join(url_transcript_text, base + '.txt')
            if os.path.isfile(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                print(f"Warning: missing transcript for {wav_file}")
                text = "-"

            entry = {
                "audio_filepath": wav_path,
                "text": text
            }

        if entry: 
            manifest_data.append(entry)

    os.makedirs(work_dir, exist_ok=True)
    manifest_path = os.path.join(work_dir, f'{output_file_name}.json')
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in manifest_data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Create manifest files for different processing types')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Common arguments for all types
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--data_dir', required=True, help='Directory containing audio files')
    base_parser.add_argument('--work_dir', required=True, help='Output directory for manifest file')
    base_parser.add_argument('--output_file', required=True, help='Base name for output manifest file')

    # ASR parser
    asr_parser = subparsers.add_parser('asr', parents=[base_parser])
    asr_parser.add_argument('--source_lang', default='en', help='Source language code') # choices=['en','de','es','fr']
    asr_parser.add_argument('--target_lang', default='en', help='Target language code') # choices=['en','de','es','fr']
    asr_parser.add_argument('--pnc', choices=['yes', 'no'], default='no', 
                          help='Whether to use punctuation and capitalization')
    asr_parser.add_argument('--include_ground_truth', action='store_true',  # NEW FLAG
                          help='Include text from .txt files')

    # Diarization parser
    diar_parser = subparsers.add_parser('diarization', parents=[base_parser])
    diar_parser.add_argument('--url_transcript_text', default=None, 
                            help='Path to transcript text file')

    # Dataset parser
    dataset_parser = subparsers.add_parser('dataset', parents=[base_parser])

    # Aligner parser
    align_parser = subparsers.add_parser('aligner', parents=[base_parser])
    align_parser.add_argument('--url_transcript_text', required=True, 
                             help='Path to transcript text file')

    args = parser.parse_args()

    # Call create_manifest with appropriate arguments
    kwargs = {
        'data_dir': args.data_dir,
        'work_dir': args.work_dir,
        'output_file_name': args.output_file,
        'manifest_type': args.command,
        'include_ground_truth': getattr(args, 'include_ground_truth', False)  # NEW        
    }

    if args.command in ['diarization', 'aligner']:
        kwargs['url_transcript_text'] = args.url_transcript_text
        
    if args.command == 'asr':
        kwargs.update({
            'source_lang': args.source_lang,
            'target_lang': args.target_lang,
            'pnc': args.pnc
        })

    create_manifest(**kwargs)

if __name__ == "__main__":
    main()
