import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import math

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

def real_time_factor(processing_time, audio_length, decimals=4):
    return None if audio_length == 0 else round(processing_time / audio_length, decimals)


def read_gold_transcription(audio_path):
    txt = Path(audio_path).with_suffix('.txt')
    return txt.read_text().strip() if txt.exists() else None


def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Phi-4-multimodal with acoustic context, chunking, and batching.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, default="transcripts", help="Where to save transcripts")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Phi-4-multimodal model")
    parser.add_argument("--generation_config", type=str, default="generation_config.json", help="Generation config JSON")
    parser.add_argument("--chunk_lengths", type=int, default=30, help="Seconds per chunk")
    parser.add_argument("--context_length", type=int, default=0, help="Seconds of acoustic context to prepend to each chunk")
    parser.add_argument("--batch_sizes", type=int, default=1, help="Number of files to process in parallel")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"], help="Audio file extensions to include")
    parser.add_argument("--gold_standard", action="store_true", default=False, help="Compute WER if gold .txt present")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor & model
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        _attn_implementation="flash_attention_2"
    ).to(device)
    generation_config = GenerationConfig.from_pretrained(args.model_path, args.generation_config)

    # Gather audio file paths
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

    main_bar = tqdm(total=total_files, desc="Batches")
    wer_bar = tqdm(total=total_files if args.gold_standard else 0, desc="WER_Calc", leave=False)

    for batch_start in range(0, total_files, args.batch_sizes):
        batch = files[batch_start : batch_start + args.batch_sizes]
        t0 = time.time()
        batch_texts = []
        batch_duration = 0.0

        # Process each file in the batch
        for path in batch:
            audio_array, sr = sf.read(path)
            total_audio_duration += len(audio_array) / sr
            batch_duration += len(audio_array) / sr

            chunk_samples = args.chunk_lengths * sr
            context_samples = args.context_length * sr
            num_chunks = math.ceil(len(audio_array) / chunk_samples)

            transcript_parts = []
            for idx in range(num_chunks):
                start = idx * chunk_samples
                end = min(len(audio_array), start + chunk_samples)
                # prepend acoustic context from previous audio
                context_start = max(0, start - context_samples)
                audio_in = audio_array[context_start:end]

                prompt = (
                    "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>"
                )
                inputs = processor(
                    text=prompt,
                    audios=[(audio_in, sr)],
                    return_tensors="pt"
                ).to(device)
                with torch.inference_mode():
                    gen_ids = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        max_new_tokens=1200
                    )
                # strip prompt tokens
                out_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(
                    out_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                transcript_parts.append(text.strip())

            batch_texts.append(" ".join(transcript_parts))

        t1 = time.time()
        # Compute and log RTF for this batch
        rtf = real_time_factor(t1 - t0, batch_duration)
        if rtf is not None:
            batch_rtfs.append(rtf)
        print(f"Batch {(batch_start // args.batch_sizes) + 1}/{(total_files + args.batch_sizes - 1)//args.batch_sizes}: RTF={rtf:.4f}")

        # Collect results and WER per file
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
                    wer_bar.update(1)
            results.append(entry)
            main_bar.update(1)

    main_bar.close()
    wer_bar.close()

    total_time = time.time() - start_all
    avg_wer = (total_wer / wer_count) if wer_count > 0 else None
    overall_rtf = real_time_factor(total_time, total_audio_duration)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = f"phi4_results_{timestamp}"
    res_file = Path(args.output_dir) / f"{out_base}.json"
    meta_file = Path(args.output_dir) / f"{out_base}_meta.json"

    with open(res_file, "w") as f_out:
        for item in results:
            f_out.write(json.dumps(item) + "\n")

    with open(meta_file, "w") as f_meta:
        json.dump({
            "total_processing_time_s": total_time,
            "audio_duration_s": total_audio_duration,
            "real_time_factor": overall_rtf,
            "batch_rtfs": batch_rtfs,
            "average_wer": avg_wer
        }, f_meta, indent=2)

    print(f"\nResults saved to: {res_file}")
    print(f"Metadata saved to: {meta_file}")
    print(f"Overall RTF={overall_rtf:.4f}" + (f" | Avg WER={avg_wer:.4f}" if avg_wer is not None else ""))

if __name__ == "__main__":
    main()
