"""
Modification of https://github.com/ponceoscarj/speech_ker/blob/main/asr/crisperwhisper_opt.py

Example:
python lite_whisper_large_v3.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/lite_whisper_large_v3 \
                --model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/lite_whisper_large_v3 \
                --chunk-lengths 30 \
                --batch-sizes 1 \
                --timestamp word \
                --extensions .wav

Notes:
- --extensions can accept multiple formats (e.g., .wav, .mp3, .flac)
- In --output_dir, insert the correct model name to stay organized.
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset, Audio
import jiwer
import warnings
from tqdm import tqdm
import time
from transformers import pipeline

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def real_time_factor(processingTime, audioLength, decimals=4):
    if audioLength == 0:
        return None
    return round(processingTime / audioLength, decimals)

def read_gold_transcription(audio_path):
    txt_path = Path(audio_path).with_suffix('.txt')
    return txt_path.read_text().strip() if txt_path.exists() else None

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

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
    parser.add_argument("--chunk-lengths", type=int, default=30,
                        help="Length of audio chunks in seconds")
    parser.add_argument("--batch-sizes", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--timestamp", choices=["word", "char", "none"], default="word",
                        help="Type of timestamps to include")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"],
                        help="Audio file extensions to process")
    parser.add_argument("--gold_standard", action="store_true", default=True,
                        help="Enable WER calculation using gold standard transcriptions")
    parser.add_argument("--sleep-time", type=int, default=0,
                        help="Optional sleep time between batches (not used currently)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    # ── Load model & processor ─────────────────────────────────────────────
    with tqdm(total=2, desc="Loading Model") as bar:
        model = AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        bar.update(1)
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
       # ── HERE ── get the special decoder prompt for Whisper
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        bar.update(1)
      
      
    # ── Load audio dataset ─────────────────────────────────────────────────
    with tqdm(total=2, desc="Loading Data") as bar:
        ds = load_dataset("audiofolder", data_dir=args.input_dir)["train"]
        bar.update(1)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        bar.update(1)

    audio_paths = [x["audio"]["path"] for x in ds]
    audio_arrays = [x["audio"]["array"] for x in ds]
    total_audio_duration = sum(len(a)/16000 for a in audio_arrays)
    total_files = len(audio_arrays)

    # ── Transcription Loop ────────────────────────────────────────────────
    main_bar = tqdm(total=total_files, desc="Transcribing", unit="file")
    wer_bar = tqdm(total=total_files, desc="WER_Calc", leave=False)
    results, batch_rtfs = [], []
    total_wer, wer_count = 0, 0
    start_all = time.time()

    for i in range(0, total_files, args.batch_sizes):
        batch_arrs = audio_arrays[i:i+args.batch_sizes]
        batch_paths = audio_paths[i:i+args.batch_sizes]
        main_bar.set_postfix(file=Path(batch_paths[0]).name)
  
        t0 = time.time()
        inputs = processor(batch_arrs, sampling_rate=16000, return_tensors="pt", padding=True).input_features
        inputs = inputs.to(device).to(dtype)
        
      with torch.inference_mode():
        pred_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

        texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        t1 = time.time()

        # compute RTF
        dur = sum(len(a)/16000 for a in batch_arrs)
        rtf = real_time_factor(t1-t0, dur)
        if rtf is not None:
            batch_rtfs.append(rtf)
        print(f"Batch {(i//args.batch_sizes)+1}: RTF={rtf:.4f}")

        # collect results + WER
        for path, text in zip(batch_paths, texts):
            entry = {"audio_file_path": path, "pred_text": text}
            if args.gold_standard:
                gold = read_gold_transcription(path)
                entry["text"] = gold or "N/A"
                if gold:
                    w = calculate_wer(gold, text)
                    entry["wer"] = w
                    total_wer += w; wer_count += 1
                    wer_bar.update(1)
            results.append(entry)
            main_bar.update(1)

        if args.sleep_time>0:
            time.sleep(args.sleep_time)

    main_bar.close(); wer_bar.close()
    total_time = time.time() - start_all

    # ── Save Outputs ───────────────────────────────────────────────────────
    rtf_all = real_time_factor(total_time, total_audio_duration)
    out_base = args.output_filename or f"results_{datetime.now().isoformat()}"
    results_file = os.path.join(args.output_dir, f"{out_base}.json")
    meta_file    = os.path.join(args.output_dir, f"{out_base}_meta.json")

    with open(results_file, "w") as f:
        for e in results:
            f.write(json.dumps(e) + "\n")
    with open(meta_file, "w") as f:
        json.dump({
            "processing_time_s": total_time,
            "audio_duration_s": total_audio_duration,
            "real_time_factor": rtf_all,
            "batch_rtfs": batch_rtfs
        }, f, indent=2)

    print(f"\nResults → {results_file}")
    print(f"Metadata → {meta_file}")
    print(f"Total Time: {total_time:.2f}s  Audio: {total_audio_duration/60:.2f}m  RTF: {rtf_all:.4f}")

if __name__ == "__main__":
    main()
