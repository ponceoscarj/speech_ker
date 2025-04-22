"""
Modification of https://github.com/ponceoscarj/speech_ker/blob/main/asr/crisperwhisper_opt.py

Example:
python lite_whisper_large_v3.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/lite_whisper_large_v3 \
                --model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/lite_whisper_large_v3 \
                --chunk_lengths 30 \
                --batch_sizes 1 \
                --timestamp segment \
                --extensions .wav

Notes:
- --extensions can accept multiple formats (e.g., .wav, .mp3, .flac)
- In --output_dir, insert the correct model name to stay organized.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModel, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import jiwer
import warnings
from tqdm import tqdm
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def real_time_factor(processingTime, audioLength, decimals=4):
    if audioLength == 0:
        return None
    return round(processingTime / audioLength, decimals)

def read_gold_transcription(audio_path):
    audio_path = Path(audio_path).resolve()
    txt_path = audio_path.with_suffix('.txt')
    if txt_path.exists():
        return txt_path.read_text().strip()
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
    parser.add_argument("--model", type=str, required=True,
                        help="Local path to the ASR model")
    parser.add_argument("--output_filename", type=str, default="",
                        help="Custom base name for output JSON file (optional)")
    parser.add_argument("--chunk_lengths", type=int, default=30,
                        help="Length of audio chunks in seconds")
    parser.add_argument("--batch_sizes", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--timestamp", choices=["word", "segment", "none"], default="segment",
                        help="Type of timestamps to include")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"],
                        help="Audio file extensions to process")
    parser.add_argument("--gold_standard", action="store_true", default=True,
                        help="Enable WER calculation using gold standard transcriptions")
    parser.add_argument("--sleep-time", type=int, default=0,
                    help="Optional sleep time between batches (not used currently)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # REMOVED: Log directory creation and log file setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    with tqdm(total=3, desc="Loading Model") as bar:
        # Load model
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True  # Allow custom modeling code
        )
        bar.update(1)
      
        if getattr(model.generation_config, "is_multilingual", False):
            model.generation_config.language = "en"
            model.generation_config.task = "transcribe"

        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        bar.update(1)

        processor.feature_extractor.return_attention_mask = True

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=args.chunk_lengths,
            batch_size=args.batch_sizes,
            return_timestamps=args.timestamp if args.timestamp != "none" else False,
            torch_dtype=torch_dtype
        )
    
    with tqdm(total=3, desc="Loading Data") as bar:
        # Parallel audio loading with memory mapping
        dataset = load_dataset("audiofolder", data_dir=args.input_dir)["train"]
        bar.update(1)
        
        # Resample to 16kHz in parallel
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        bar.update(1)

        # Prepare file paths
        audio_paths = [str(Path(x['audio']['path']).resolve()) for x in dataset]
        bar.update(1)

        audio_arrays = [x['audio']['array'] for x in dataset]

        # Calculate total audio duration
        total_audio_duration = sum([len(x['audio']['array']) / x['audio']['sampling_rate'] for x in dataset])

    total_files = len(dataset)
    main_bar = tqdm(total=total_files, desc="Transcribing", unit="file")
    wer_bar = tqdm(total=total_files, desc="WER_calculation", leave=False)

    results = []
    total_wer = 0
    valid_wer_count = 0
    start_time = time.time()
    batch_rtf_list = []

    try:
        for i in range(0, len(dataset), args.batch_sizes):
            batch = dataset[i:i+args.batch_sizes]
            batch_audio_arrays = audio_arrays[i:i+args.batch_sizes]
            batch_paths = audio_paths[i:i+args.batch_sizes]

            main_bar.set_postfix(file=os.path.basename(batch_paths[0]))

            batch_start_time = time.time()

            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    outputs = pipe(batch_audio_arrays)

            batch_end_time = time.time()
            batch_processing_time = batch_end_time - batch_start_time
            batch_audio_duration = sum([len(arr) / 16000 for arr in batch_audio_arrays])

            batch_rtf = real_time_factor(batch_processing_time, batch_audio_duration)

            if batch_rtf is not None:
                print(f"Batch {i // args.batch_sizes + 1}: Processing Time = {batch_processing_time:.2f} sec, RTF = {batch_rtf:.4f}")
                batch_rtf_list.append(batch_rtf)
            else:
                print(f"Batch {i // args.batch_sizes + 1}: Audio duration zero, cannot calculate RTF.")

            for path, result in zip(batch_paths, outputs):
                pred_text = result["text"] if isinstance(result, dict) else result
                entry = {
                    "audio_file_path": path,
                    "pred_text": pred_text
                }

                if args.gold_standard:
                    gold_text = read_gold_transcription(path)
                    entry["text"] = gold_text or "N/A"
                    
                    if gold_text:
                        wer = calculate_wer(entry["text"], entry["pred_text"])
                        entry["wer"] = wer
                        total_wer += wer
                        valid_wer_count += 1
                        wer_bar.update(1)
                        wer_bar.set_postfix(current_wer=f"{wer:.2f}")

                if args.gold_standard and gold_text:
                    print(f'Processed {args.batch_sizes}. WER = {wer:.2f}')
                else:
                    print(f'Processed {args.batch_sizes}.')

                results.append(entry)
                main_bar.update(1)

    finally:
        main_bar.close()
        wer_bar.close()

    end_time = time.time()
    processing_time = end_time - start_time

    # ====================== Save Results ======================
    rtf = real_time_factor(processing_time, total_audio_duration)
    total_audio_minutes = total_audio_duration / 60

    if args.output_filename:
        results_file = os.path.join(args.output_dir, f"{args.output_filename}.json")
        meta_file = os.path.join(args.output_dir, f"{args.output_filename}_meta.json")
    else:
        results_file = os.path.join(args.output_dir, f"results_{datetime.now().isoformat()}.json")
        meta_file = os.path.join(args.output_dir, f"results_{datetime.now().isoformat()}_meta.json")

    print('\nTranscription Script\n', 'input_dir', args.input_dir)
    print('results_file', results_file, '\n')

    with open(results_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    meta = {
        "processing_time_seconds": processing_time,
        "total_audio_duration_seconds": total_audio_duration,
        "real_time_factor": rtf,
        "batch_rtf_list": batch_rtf_list
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Transcriptions saved to {results_file}")
    print(f"Run metadata saved to   {meta_file}")
    print(f"Total Processing Time: {processing_time:.2f} seconds")
    print(f"Total Audio Duration: {total_audio_minutes:.2f} minutes")

    if rtf is not None:
        print(f"\nReal-Time Factor (RTF): {rtf}")
    else:
        print("\nWarning: Total audio duration is zero, cannot calculate RTF.")

    if args.gold_standard and valid_wer_count > 0:
        print(f"\nAverage WER: {total_wer/valid_wer_count:.2f}")
    print(f"\nProcessing complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n[INFO] GPU cache cleared successfully.")
