
'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python crisperwhisper.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/CrisperWhisper \
                --model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/CrisperWhisper \
                --chunk_length 30 \
                --batch_size 1 \
                --timestamps none \
                --
                --extensions .wav


Notes:
--extensions .wav . mp3 #can accept multiple
In --output_dir insert the correct model, be as specific as possible (e.g., canary-1b, canary-1b-flash, canary-180m)
'''
import argparse
import json
import os
import sys
import torch
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import jiwer
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def read_gold_transcription(audio_path):
    base_name = os.path.splitext(audio_path)[0]
    txt_path = base_name + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return f.read().strip()
    return None

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

# REMOVED: Tee class and log file handling

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="transcripts",
                       help="Output directory for transcriptions")
    parser.add_argument("--model", type=str, default="nyrahealth/CrisperWhisper",
                       help="ASR model identifier from Hugging Face Hub or local path")
    parser.add_argument("--output_filename", type=str, default="",
                       help="Custom base name for output JSON file (optional)")    
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

    # REMOVED: Log directory creation and log file setup

    try:
        # CHANGED: Simplified startup messages
        print(f"\n=== Transcription form crisperwhisper.py Started ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Command: {' '.join(sys.argv)}")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using device: {'GPU' if 'cuda' in device else 'CPU'}")
        print(f"Output directory: {args.output_dir}\n")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            attn_implementation="sdpa" if torch.cuda.is_available() else None            
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
            device=device,
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
                "return_timestamps": "none"}            
        )

        audio_files = [
            f for f in os.listdir(args.input_dir)
            if os.path.splitext(f)[1].lower() in args.extensions
        ]

        if not audio_files:
            print(f"Error: No audio files found with extensions {args.extensions} in {args.input_dir}")
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
                            print(f"Processed {audio_file} - WER: {wer:.2f}")
                        else:
                            entry["wer"] = "N/A"
                            print(f"Processed {audio_file} - No gold standard")
                    else:
                        print(f"Processed {audio_file}")

                    results.append(entry)

                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
                    entry["error"] = str(e)
                    results.append(entry)

        filename_components = [args.output_filename, f"{args.batch_size}", f"{args.chunk_length}"]

        json_name = "_".join(filename_components) + ".json"
        output_json = os.path.join(args.output_dir, json_name)        
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)

        if args.gold_standard and valid_wer_count > 0:
            avg_wer = total_wer / valid_wer_count
            print(f"\n=== Summary ===")
            print(f"Average Word Error Rate: {avg_wer:.4f}")
            print(f"Processed files: {valid_wer_count}")
        else:
            print("\n=== Completed ===")
            print(f"Processed files: {len(audio_files)}")

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()