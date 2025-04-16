import argparse
import json
import os
import sys
import torch
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import jiwer
'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python crisperwhisper.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/CrisperWhisper \
                --model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/CrisperWhisper \
                --chunk_length 30 \
                --batch_size 1 \
                --timestamps none \
                --extensions .wav


Notes:
--extensions .wav . mp3 #can accept multiple
In --output_dir insert the correct model, be as specific as possible (e.g., canary-1b, canary-1b-flash, canary-180m)
'''

def read_gold_transcription(audio_path):
    base_name = os.path.splitext(audio_path)[0]
    txt_path = base_name + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return f.read().strip()
    return None

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

class Tee:
    """Duplicate output to both console and log file"""
    def __init__(self, *files):
        self.files = files
        
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
            
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="transcripts",
                       help="Output directory for transcriptions")
    parser.add_argument("--model", type=str, default="nyrahealth/CrisperWhisper",
                       help="ASR model identifier from Hugging Face Hub or local path")
    parser.add_argument("--chunk_length", type=int, default=30,
                     help="Length of audio chunks in seconds")
    parser.add_argument("--batch_size", type=int, default=1,
                     help="Batch size for processing")
    parser.add_argument("--timestamps", choices=["word", "segment", "none"], default="word",
                     help="Type of timestamps to include")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"],
                     help="Audio file extensions to process")
    parser.add_argument("--gold_standard", action="store_true",
                       help="Enable WER calculation using gold standard transcriptions")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"transcription_log_{timestamp}.txt")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_file, 'w') as f:
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)

        try:
            print(f"Transcription Log - {timestamp}")
            print(f"Command: {' '.join(sys.argv)}")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device_type = "GPU" if "cuda" in device else "CPU"
            print(f"Using device: {device_type}")
            print(f"Output directory: {args.output_dir}\n")

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(args.model)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=args.chunk_length,
                batch_size=args.batch_size,
                return_timestamps=args.timestamps if args.timestamps != "none" else False,
                torch_dtype=torch_dtype,
                device=device
            )

            audio_files = [
                f for f in os.listdir(args.input_dir)
                if os.path.splitext(f)[1].lower() in args.extensions
            ]

            if not audio_files:
                print(f"No audio files found with extensions {args.extensions} in {args.input_dir}")
                sys.exit(1)

            results = []
            total_wer = 0
            valid_wer_count = 0

            batch_size = args.batch_size
            audio_batches = [audio_files[i:i + batch_size] for i in range(0, len(audio_files), batch_size)]

            for batch in audio_batches:
                batch_file_paths = [os.path.join(args.input_dir, f) for f in batch]
                try:
                    batch_results = pipe(batch_file_paths)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    for audio_file in batch:
                        entry = {
                            "audio_file_path": os.path.join(args.input_dir, audio_file),
                            "error": str(e)
                        }
                        results.append(entry)
                    continue

                for audio_file, result in zip(batch, batch_results):
                    entry = {"audio_file_path": os.path.join(args.input_dir, audio_file)}
                    try:
                        predicted_text = result.get("text", "")
                        entry["predicted_transcription"] = predicted_text

                        if args.gold_standard:
                            gold_text = read_gold_transcription(entry["audio_file_path"])
                            entry["gold_transcription"] = gold_text if gold_text else "N/A"
                            if gold_text:
                                wer = calculate_wer(gold_text, predicted_text)
                                entry["wer"] = wer
                                total_wer += wer
                                valid_wer_count += 1
                            else:
                                entry["wer"] = "N/A"

                        results.append(entry)
                        print(f"Processed {audio_file}")

                    except Exception as e:
                        print(f"Error processing {audio_file}: {str(e)}")
                        entry["error"] = str(e)
                        results.append(entry)

            output_json = os.path.join(args.output_dir, "results.json")
            with open(output_json, "w") as f:
                json.dump(results, f, indent=4)

            if args.gold_standard and valid_wer_count > 0:
                avg_wer = total_wer / valid_wer_count
                print(f"\nAverage Word Error Rate: {avg_wer:.4f} (calculated over {valid_wer_count} files)")
            else:
                print("\nNo valid WER calculations were possible - missing gold standards")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()
