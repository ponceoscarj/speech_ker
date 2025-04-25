import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import soundfile as sf
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig
)
import jiwer
import warnings
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Color constants
COLOR_FILE = "#00ff00"
COLOR_BATCH = "#ffff00"
COLOR_CHUNK = "#00ffff"
COLOR_WER = "#ff00ff"

def real_time_factor(processing_time, audio_length, decimals=4):
    return None if audio_length == 0 else round(processing_time / audio_length, decimals)

def read_gold_transcription(audio_path):
    txt_path = Path(audio_path).with_suffix('.txt')
    return txt_path.read_text().strip() if txt_path.exists() else None

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

def chunk_audio(array, sample_rate, chunk_sec):
    chunk_size = int(chunk_sec * sample_rate)
    return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]

def format_time(seconds):
    return f"{seconds:.1f}s"

def main():
    parser = argparse.ArgumentParser(description="True batch inference with chunk processing")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="transcripts")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="")
    parser.add_argument("--generation_config", type=str, default="generation_config.json")
    parser.add_argument("--chunk_lengths", type=int, default=30)
    parser.add_argument("--batch_sizes", type=int, default=4, help="Number of chunks to process in parallel")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"])
    parser.add_argument("--gold_standard", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model components
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2"
    ).to(device)
    generation_config = GenerationConfig.from_pretrained(args.model_path, args.generation_config)

    # File handling
    files = []
    for ext in args.extensions:
        files.extend(Path(args.input_dir).rglob(f"*{ext}"))
    files = sorted(files)
    total_files = len(files)

    results = []
    batch_rtfs = []
    total_wer, wer_count = 0.0, 0
    total_audio_duration = 0.0
    start_all = time.time()

    # Main progress bar
    with tqdm(total=total_files, desc="📁 Files", colour=COLOR_FILE, position=0) as main_bar:
        # File batch processing
        for batch_idx, batch_start in enumerate(range(0, total_files, args.batch_sizes), 1):
            file_batch = files[batch_start:batch_start + args.batch_sizes]
            batch_files = [Path(p).name for p in file_batch]
            
            # Collect all chunks from all files in this batch
            all_chunks = []
            chunk_metadata = []
            for path in file_batch:
                audio_array, sr = sf.read(path)
                duration = len(audio_array) / sr
                total_audio_duration += duration
                chunks = chunk_audio(audio_array, sr, args.chunk_lengths)
                all_chunks.extend([(chunk, sr) for chunk in chunks])
                chunk_metadata.extend([(path, duration)] * len(chunks))

            # Batch process all chunks
            total_chunks = len(all_chunks)
            tqdm.write(f"\n🚀 Processing {len(file_batch)} files ({total_chunks} chunks)")
            
            batch_texts = defaultdict(list)
            with tqdm(total=total_chunks, desc="🔊 Chunks", colour=COLOR_CHUNK, 
                     position=1, leave=False) as chunk_bar:
                # Process in chunk batches
                for chunk_start in range(0, total_chunks, args.batch_sizes):
                    chunk_batch = all_chunks[chunk_start:chunk_start + args.batch_sizes]
                    
                    # Prepare batch inputs
                    inputs = processor(
                        text=["<|user|><|audio_1|>...<|end|><|assistant|>"] * len(chunk_batch),
                        audios=chunk_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)

                    # Generate text
                    with torch.inference_mode():
                        gen_ids = model.generate(
                            **inputs,
                            generation_config=generation_config,
                            max_new_tokens=1200
                        )

                    # Decode outputs
                    outputs = processor.batch_decode(
                        gen_ids[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    # Map outputs back to files
                    for idx, output in enumerate(outputs):
                        original_path = chunk_metadata[chunk_start + idx][0]
                        batch_texts[original_path].append(output.strip())
                    
                    chunk_bar.update(len(chunk_batch))

            # Combine chunk results per file
            for path in file_batch:
                transcript = " ".join(batch_texts[path])
                results.append({
                    "audio_file_path": str(path),
                    "pred_text": transcript
                })
                
                # WER calculation
                if args.gold_standard:
                    gold = read_gold_transcription(str(path))
                    if gold:
                        wer = calculate_wer(gold, transcript)
                        total_wer += wer
                        wer_count += 1
                        main_bar.set_postfix_str(f"Avg WER: {total_wer/wer_count:.2%}")

            main_bar.update(len(file_batch))

    # Save results and metadata
    out_base = args.output_filename or f"results_{datetime.now().isoformat()}"
    res_file = Path(args.output_dir) / f"{out_base}.json"
    meta_file = Path(args.output_dir) / f"{out_base}_meta.json"
    
    with open(res_file, "w") as f:
        json.dump(results, f, indent=2)
        
    total_time = time.time() - start_all
    meta_data = {
        "total_time": total_time,
        "audio_duration": total_audio_duration,
        "rtf": real_time_factor(total_time, total_audio_duration),
        "avg_wer": total_wer / wer_count if wer_count else None
    }
    
    with open(meta_file, "w") as f:
        json.dump(meta_data, f, indent=2)

    print(f"\n✅ Processing complete! RTF: {meta_data['rtf']:.2f}")

if __name__ == "__main__":
    main()