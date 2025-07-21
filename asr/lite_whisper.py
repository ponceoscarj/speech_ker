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
import soundfile as sf

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
     # Normalized transformation
    norm_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveMultipleSpaces()
    ])

    # Raw WER
    raw_measures = jiwer.compute_measures(reference, hypothesis)

    # Normalized WER
    norm_measures = jiwer.compute_measures(reference, hypothesis,
                                           truth_transform=norm_transform,
                                           hypothesis_transform=norm_transform)

    return raw_measures, norm_measures
  
def chunk_audio(audio_array, sr, chunk_sec):
    chunk_size = int(sr * chunk_sec)
    return [audio_array[i:i + chunk_size] for i in range(0, len(audio_array), chunk_size)]

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="transcripts",
                        help="Output directory for transcriptions")
    parser.add_argument("--main_model", type=str, required=True,
                        help="Local path to the ASR model")
    parser.add_argument("--processor_model", type=str, default="openai/whisper-large-v3",
                        help="Local path for the Whisper processor")
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
            args.main_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        
        bar.update(1)
        processor = AutoProcessor.from_pretrained(args.processor_model)
      
       # ── HERE ── get the special decoder prompt for Whisper
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        bar.update(1)
    
  # ── Transcription Loop ────────────────────────────────────────────────
    files = []
    for ext in args.extensions:
        files.extend(Path(args.input_dir).rglob(f"*{ext}"))
    files = sorted(files)
    total_files = len(files)
    
    main_bar = tqdm(total=total_files, desc="Transcribing", unit="file")
    wer_bar = tqdm(total=total_files, desc="WER_Calc", leave=False)
    results, batch_rtfs = [], []
    total_rwer, total_nwer = 0.0, 0.0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    wer_count = 0
    total_audio_duration = 0.0
    start_all = time.time()
    
    for batch_start in range(0, total_files, args.batch_sizes):
        batch_files = files[batch_start:batch_start + args.batch_sizes]
        chunks_by_file = {}
        srs = {}
    
        # Read and chunk all files in batch
        for path in batch_files:
            audio, sr = sf.read(path)
            srs[path] = sr
            duration = len(audio) / sr
            total_audio_duration += duration
            chunk_len = int(args.chunk_lengths * sr)
            chunks = [audio[i:i+chunk_len] for i in range(0, len(audio), chunk_len)]
            chunks_by_file[path] = chunks
    
        max_rounds = max(len(chunks) for chunks in chunks_by_file.values())
        batch_texts = {path: [] for path in batch_files}
    
        with tqdm(total=max_rounds, desc="🔊 Chunk rounds", leave=False) as chunk_round_bar:
            for round_idx in range(max_rounds):
                to_process = []
                paths_order = []
                for path in batch_files:
                    if round_idx < len(chunks_by_file[path]):
                        to_process.append(chunks_by_file[path][round_idx])
                        paths_order.append(path)
    
                t0 = time.time()
                inputs = processor(
                    to_process,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).to(device)
    
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    pred_ids = model.generate(
                        inputs.input_features,
                        forced_decoder_ids=forced_decoder_ids
                    )
                    outputs = processor.batch_decode(pred_ids, skip_special_tokens=True)
                t1 = time.time()
    
                for path, text in zip(paths_order, outputs):
                    batch_texts[path].append(text.strip())
    
                dur = sum(len(chunk) / 16000 for chunk in to_process)
                rtf = real_time_factor(t1 - t0, dur)
                if rtf is not None:
                    batch_rtfs.append(rtf)
                print(f"Batch {(batch_start // args.batch_sizes) + 1}, Round {round_idx + 1}: RTF={rtf:.4f}")
                chunk_round_bar.update(1)
    
        for path in batch_files:
            full_text = " ".join(batch_texts[path])
            entry = {"audio_file_path": str(path), "pred_text": full_text}
            if args.gold_standard:
                gold = read_gold_transcription(path)
                entry["text"] = gold or "N/A"
                if gold:
                    raw_measures, norm_measures = calculate_wer_breakdown(gold, full_text)
                    entry.update({
                      # Raw metrics
                      "rWER": raw_measures["wer"],
                      "rInsertions": raw_measures["insertions"],
                      "rDeletions": raw_measures["deletions"],
                      "rSubstitutions": raw_measures["substitutions"],
                      "rHits": raw_measures["hits"],
                      "rRefLen": raw_measures["truth_length"],
                
                      # Normalized metrics
                      "nWER": norm_measures["wer"],
                      "nInsertions": norm_measures["insertions"],
                      "nDeletions": norm_measures["deletions"],
                      "nSubstitutions": norm_measures["substitutions"],
                      "nHits": norm_measures["hits"],
                      "nRefLen": norm_measures["truth_length"]
                    })
                  
                    total_insertions += raw_measures["insertions"]
                    total_deletions += raw_measures["deletions"]
                    total_substitutions += raw_measures["substitutions"]
                    total_rwer += raw_measures["wer"]
                    total_nwer += norm_measures["wer"]
                    wer_count += 1
                    wer_bar.update(1)
                  
            results.append(entry)
            main_bar.update(1)
    
        if args.sleep_time > 0:
            time.sleep(args.sleep_time)
    
    main_bar.close()
    wer_bar.close()
    total_time = time.time() - start_all
    
    if args.gold_standard and wer_count > 0:
        avg_rwer = total_rwer / wer_count
        avg_nwer = total_nwer / wer_count
        avg_insertions = total_insertions / wer_count
        avg_deletions = total_deletions / wer_count
        avg_substitutions = total_substitutions / wer_count
    
        print("\n===== WER Summary =====")
        print(f"Average Normalized WER:  {avg_nwer:.2f}")
        print(f"Average Raw WER:         {avg_rwer:.2f}")
        print(f"Average Insertions:      {avg_insertions:.2f}")
        print(f"Average Deletions:       {avg_deletions:.2f}")
        print(f"Average Substitutions:   {avg_substitutions:.2f}")
    
      
    # ── Save Outputs ───────────────────────────────────────────────────────
    rtf_all = real_time_factor(total_time, total_audio_duration)
    out_base = args.output_filename or f"results_{datetime.now().isoformat()}"
    results_file = os.path.join(args.output_dir, f"{out_base}.json")
    meta_file = os.path.join(args.output_dir, f"{out_base}_meta.json")
    
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
    print(f"Total Time: {total_time:.2f}s  Audio: {total_audio_duration / 60:.2f}m  RTF: {rtf_all:.4f}")

if __name__ == "__main__":
    main()
