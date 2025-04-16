
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
from datasets import load_dataset, Audio
import jiwer
import warnings
from tqdm import tqdm

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32


    with tqdm(total=3, desc="Loading Model") as bar:
        # Flash Attention 2 for 3-5x speedup
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            # use_flash_attention_2=True,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        bar.update(1)

        processor = AutoProcessor.from_pretrained(args.model)
        bar.update(1)

        processor.feature_extractor.return_attention_mask = True

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=args.chunk_length,
            batch_size=args.batch_size,
            return_timestamps=args.timestamps if args.timestamps != "none" else False,
            torch_dtype=torch_dtype,
            device=device)
        
    with tqdm(total=3, desc="Loading Data") as bar:
        # Parallel audio loading with memory mapping
        dataset = load_dataset("audiofolder", data_dir=args.input_dir)["train"]
        bar.update(1)
        
        # Resample to 16kHz in parallel
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        bar.update(1)
        
        # Prepare file paths
        audio_paths = [os.path.join(args.input_dir, x['audio']['path'].split('/')[-1]) for x in dataset]
        bar.update(1)        
    
    total_files = len(dataset)
    main_bar = tqdm(total=total_files, desc="Transcribing", unit="file")
    wer_bar = tqdm(total=total_files, desc="WER_calculation", leave=False)

    results = []
    total_wer = 0
    valid_wer_count = 0

    try:
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i:i+args.batch_size]
            batch_paths = audio_paths[i:i+args.batch_size]
            
            # Show current file being processed
            main_bar.set_postfix(file=os.path.basename(batch_paths[0]))
            
            # Batch inference
            outputs = pipe([x["array"] for x in batch["audio"]])
            
            # Process results
            for path, result in zip(batch_paths, outputs):
                entry = {
                    "audio_file_path": path,
                    "predicted_transcription": result.get("text", "")
                }
                
                # WER calculation
                if args.gold_standard:
                    gold_text = read_gold_transcription(path)
                    entry["gold_transcription"] = gold_text or "N/A"
                    print(gold_text)
                    
                    if gold_text:
                        wer = calculate_wer(gold_text, entry["predicted_transcription"])
                        entry["wer"] = wer
                        total_wer += wer
                        valid_wer_count += 1
                        wer_bar.update(1)
                        wer_bar.set_postfix(current_wer=f"{wer:.2f}")
                print(f'Processed {args.batch_size}. WER = {wer}')
                results.append(entry)
                main_bar.update(1)

    finally:
        main_bar.close()
        wer_bar.close()

    # ====================== Save Results ======================
    output_file = os.path.join(args.output_dir, f"results_{datetime.now().isoformat()}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    if args.gold_standard and valid_wer_count > 0:
        print(f"\nAverage WER: {total_wer/valid_wer_count:.2f}")
    print(f"\nProcessing complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()