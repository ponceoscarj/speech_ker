import os
import argparse
import torch
from tqdm import tqdm
from torch.torch_version import TorchVersion
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution

'''
Example

python script.py \
    --input /path/to/audio_files \
    --output /path/to/output \
    --token YOUR_HF_TOKEN \
    [--gpu]

YOUR_HF_TOKEN is your access token from huggingface
'''

def main():
    # Configure safe globals
    torch.serialization.add_safe_globals([TorchVersion, Specifications, Problem, Resolution])

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Speaker Diarization Processor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                      help='Input folder containing audio files')
    parser.add_argument('-o', '--output', required=True,
                      help='Output folder for text results')
    parser.add_argument('-t', '--token', required=True,
                      help='Hugging Face auth token')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU acceleration if available')
    
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=args.token
    )

    if args.gpu and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    def is_audio_file(filename):
        audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.aiff'}
        return os.path.splitext(filename)[1].lower() in audio_extensions

    # Get list of audio files
    audio_files = [f for f in os.listdir(args.input) if is_audio_file(f)]
    
    # Process files with progress bar
    for filename in tqdm(audio_files, desc="Processing audio files"):
        input_path = os.path.join(args.input, filename)
        output_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}.txt")
        
        try:
            diarization = pipeline(input_path)
            with open(output_path, 'w') as f:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    f.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}\n")
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()