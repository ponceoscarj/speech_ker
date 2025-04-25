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
COLOR_CHUNK = "#00ffff"
COLOR_WER = "#ff00ff"

PROMPT = (
    "<|user|><|audio_1|>Based on the attached audio, generate a comprehensive text transcription"
    " of the spoken content.<|end|><|assistant|>"
)

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

def main():
    parser = argparse.ArgumentParser(description="Per-file batch inference with chunk processing")
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

    # Load model
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2"
    ).to(device)

    model._attn_implementation = "flash_attention_2"

    generation_config = GenerationConfig.from_pretrained(args.model_path, args.generation_config)

    # Gather files
    files = []
    for ext in args.extensions:
        files.extend(Path(args.input_dir).rglob(f"*{ext}"))
    files = sorted(files)
    total_files = len(files)

    results = []
    total_wer, wer_count = 0.0, 0
    total_audio_duration = 0.0
    start_all = time.time()

    # File-level progress
    with tqdm(total=total_files, desc="📁 Files", colour=COLOR_FILE, position=0) as file_bar:
        # Iterate files in file-batches
        for batch_start in range(0, total_files, args.batch_sizes):
            batch_files = files[batch_start:batch_start + args.batch_sizes]
            # Prepare per-file chunks
            chunks_by_file = {}
            srs = {}
            for path in batch_files:
                audio, sr = sf.read(path)
                srs[path] = sr
                duration = len(audio) / sr
                total_audio_duration += duration
                chunks_by_file[path] = chunk_audio(audio, sr, args.chunk_lengths)

            # Determine number of rounds (max chunks in batch)
            max_rounds = max(len(chunks) for chunks in chunks_by_file.values())
            batch_texts = {path: [] for path in batch_files}

            # Chunk-round progress bar
            with tqdm(total=max_rounds, desc="🔊 Chunk rounds", colour=COLOR_CHUNK, position=1, leave=False) as chunk_round_bar:
                # For each chunk index
                for i in range(max_rounds):
                    # Gather up to batch_sizes chunks from each file
                    to_process = []
                    paths_order = []
                    for path in batch_files:
                        chunks = chunks_by_file[path]
                        if i < len(chunks):
                            to_process.append((chunks[i], srs[path]))
                            paths_order.append(path)

                    if not to_process:
                        break

                    # Process this sub-batch of chunks
                    inputs = processor(
                        text=[PROMPT] * len(to_process),
                        audios=to_process,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)

                    with torch.inference_mode():
                        gen_ids = model.generate(
                            **inputs,
                            generation_config=generation_config,
                            max_new_tokens=1200
                        )

                    outputs = processor.batch_decode(
                        gen_ids[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    # Assign back to each file
                    for j, text in enumerate(outputs):
                        batch_texts[paths_order[j]].append(text.strip())

                    chunk_round_bar.update(1)

            # Finalize each file
            for path in batch_files:
                transcript = " ".join(batch_texts[path])
                entry = {"audio_file_path": str(path), "pred_text": transcript}
                if args.gold_standard:
                    gold = read_gold_transcription(str(path))
                    if gold:
                        wer = calculate_wer(gold, transcript)
                        total_wer += wer
                        wer_count += 1
                        file_bar.set_postfix({"Avg WER": f"{(total_wer/wer_count):.2%}"})
                        entry["wer"] = wer
                results.append(entry)
                file_bar.update(1)

    # Save output
    out_base = args.output_filename or f"results_{datetime.now().isoformat()}"
    res_file = Path(args.output_dir) / f"{out_base}.json"
    meta_file = Path(args.output_dir) / f"{out_base}_meta.json"

    with open(res_file, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_all
    meta = {
        "total_time": total_time,
        "audio_duration": total_audio_duration,
        "rtf": real_time_factor(total_time, total_audio_duration),
        "avg_wer": (total_wer/wer_count) if wer_count else None
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Processing complete! RTF: {meta['rtf']:.2f}")

if __name__ == "__main__":
    main()
