import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

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

# Color constants for progress bars
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
    parser = argparse.ArgumentParser(description="Transcribe audio using Phi-4-multimodal with chunking and batching.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, default="transcripts", help="Where to save transcripts")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Phi-4-multimodal model")
    parser.add_argument("--output_filename", type=str, default="",
                        help="Custom base name for output JSON file (optional)")
    parser.add_argument("--generation_config", type=str, default="generation_config.json", help="Generation config JSON")
    parser.add_argument("--chunk_lengths", type=int, default=30, help="Seconds per chunk")
    parser.add_argument("--batch_sizes", type=int, default=1, help="Number of files to process in parallel")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"], help="Audio file extensions to include")
    parser.add_argument("--gold_standard", action="store_true", default=False, help="Compute WER if gold .txt present")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor & model
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        _attn_implementation=None
    ).to(device)
    model._attn_implementation = "flash_attention_2"
    generation_config = GenerationConfig.from_pretrained(args.model_path, args.generation_config)

    # Gather audio file paths
    files = []
    for ext in args.extensions:
        files.extend(Path(args.input_dir).rglob(f"*{ext}"))
    files = sorted(files)
    total_files = len(files)

    # Initialize progress tracking
    results = []
    batch_rtfs = []
    total_wer, wer_count = 0.0, 0
    total_audio_duration = 0.0
    start_all = time.time()

    # Main progress bar
    with tqdm(
        total=total_files,
        desc="📁 Overall Progress",
        unit="file",
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour=COLOR_FILE,
        position=0
    ) as main_bar:

        # Process in batches
        for batch_idx, batch_start in enumerate(range(0, total_files, args.batch_sizes), 1):
            batch = files[batch_start : batch_start + args.batch_sizes]
            batch_size = len(batch)
            batch_duration = 0.0
            batch_texts = []
            tqdm.write(f"\n🚀 Starting Batch {batch_idx} ({batch_size} files)")

            # Batch timer
            batch_timer = tqdm(
                total=None,
                desc=f"⚡ Processing Batch {batch_idx}",
                bar_format="{desc}: {elapsed}",
                colour=COLOR_BATCH,
                position=1,
                leave=False
            )

            # Process each file in batch
            for path in batch:
                file_timer = time.time()
                audio_array, sr = sf.read(path)
                duration = len(audio_array) / sr
                total_audio_duration += duration
                batch_duration += duration

                # Chunk processing
                chunks = chunk_audio(audio_array, sr, args.chunk_lengths)
                transcript_parts = []
                
                with tqdm(
                    chunks,
                    desc=f"🔊 {Path(path).name[:15]}...",
                    unit="chunk",
                    colour=COLOR_CHUNK,
                    position=2,
                    leave=False,
                    bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                ) as chunk_bar:
                    for chunk in chunk_bar:
                        prompt = "<|user|><|audio_1|>...<|end|><|assistant|>"
                        inputs = processor(text=prompt, audios=[(chunk, sr)], return_tensors="pt").to(device)
                        with torch.inference_mode():
                            gen_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1200)
                        out_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
                        text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        transcript_parts.append(text.strip())
                
                batch_texts.append(" ".join(transcript_parts))
                main_bar.update(1)
                file_time = time.time() - file_timer
                tqdm.write(f"   ✅ {Path(path).name} processed in {format_time(file_time)}")

            # Finish batch processing
            batch_time = time.time() - batch_timer.last_print_t
            rtf = real_time_factor(batch_time, batch_duration)
            batch_rtfs.append(rtf or 0)
            batch_timer.close()
            tqdm.write(f"🏁 Batch {batch_idx} completed | RTF: {rtf:.2f} | Time: {format_time(batch_time)}")

            # Process results
            for path, hyp in zip(batch, batch_texts):
                entry = {"audio_file_path": str(path), "pred_text": hyp}
                if args.gold_standard:
                    gold = read_gold_transcription(str(path))
                    entry["text"] = gold or "N/A"
                    if gold:
                        w = calculate_wer(gold, hyp)
                        entry["wer"] = w
                        total_wer += w
                        wer_count += 1
                        main_bar.set_postfix_str(f"Avg WER: {total_wer/wer_count:.2%}" if wer_count else "")
                results.append(entry)

    # Final calculations
    total_time = time.time() - start_all
    avg_wer = (total_wer / wer_count) if wer_count > 0 else None
    overall_rtf = real_time_factor(total_time, total_audio_duration)

    # Save outputs
    out_base = args.output_filename or f"results_{datetime.now().isoformat()}"
    res_file = Path(args.output_dir) / f"{out_base}.json"
    meta_file = Path(args.output_dir) / f"{out_base}_meta.json"

    with open(res_file, "w") as f_out:
        json.dump(results, f_out, indent=2)

    meta_data = {
        "total_processing_time_s": total_time,
        "audio_duration_s": total_audio_duration,
        "real_time_factor": overall_rtf,
        "batch_rtfs": batch_rtfs,
        "average_wer": avg_wer
    }
    with open(meta_file, "w") as f_meta:
        json.dump(meta_data, f_meta, indent=2)

    print(f"\n🎉 Processing complete!")
    print(f"📄 Results saved to: {res_file}")
    print(f"📊 Metadata saved to: {meta_file}")
    print(f"⏱️ Overall RTF: {overall_rtf:.2f}" + (f" | 🎯 Avg WER: {avg_wer:.2%}" if avg_wer else ""))

if __name__ == "__main__":
    main()