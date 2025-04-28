#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import WhisperForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio
from tqdm import tqdm
import jiwer

# === Settings ===
MODEL_NAME = "openai/whisper-large-v3"
INPUT_DIR = "/home/ext_alzahidy_misk_mayo_edu/speech_ker/audio_files/audio_valid"
OUTPUT_DIR = "/home/ext_alzahidy_misk_mayo_edu/speech_ker/asr/output/whisper-large-v3-sequential"
OUTPUT_FILENAME = f"results_{datetime.now().isoformat()}.json"

BATCH_SIZE = 1  # safe start; increase later if memory allows
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Step 1: Load Model and Processor ===
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)

# === Step 2: Load your audio files ===
print("Loading audio files...")
dataset = load_dataset(
    "audiofolder",
    data_dir=INPUT_DIR,
    split="train",
    trust_remote_code=True,
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Prepare audios
audio_arrays = [x['audio']['array'].astype(np.float32) for x in dataset]
audio_paths = [str(Path(x['audio']['path']).resolve()) for x in dataset]

# === Step 3: Prepare Inputs ===
print("Processing inputs...")
inputs = processor(
    audio_arrays,
    return_tensors="pt",
    padding="longest",
    truncation=False,
    sampling_rate=16000,
    return_attention_mask=True,
)

inputs = {k: v.to(DEVICE, torch.float16) for k, v in inputs.items()}

# === Step 4: Generate Transcriptions ===
print("Generating transcriptions...")

generated_tokens = model.generate(
    **inputs,
    condition_on_prev_tokens=False,
    return_timestamps=True,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold=-1.0,
    compression_ratio_threshold=1.35,
    max_length=448,
    no_repeat_ngram_size=0
)

# === Step 5: Decode Outputs ===
print("Decoding results...")
decoded_texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)

# === Step 6: Save Transcriptions and Meta ===
print("Saving outputs...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
meta_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME.replace(".json", "_meta.json"))

results = []
wer_scores = []
total_audio_duration_sec = 0

def read_gold_transcription(audio_path):
    audio_path = Path(audio_path)
    txt_path = audio_path.with_suffix('.txt')
    if txt_path.exists():
        return txt_path.read_text().strip()
    else:
        return None

start_time = datetime.now()

for path, pred_text in tqdm(zip(audio_paths, decoded_texts), total=len(decoded_texts), desc="Saving results"):
    entry = {
        "audio_file_path": path,
        "pred_text": pred_text,
    }
    
    # Try to find gold text
    gold_text = read_gold_transcription(path)
    if gold_text:
        entry["gold_text"] = gold_text
        wer = jiwer.wer(gold_text, pred_text)
        entry["wer"] = wer
        wer_scores.append(wer)

    results.append(entry)

# Save JSON results
with open(results_file, "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

end_time = datetime.now()
processing_seconds = (end_time - start_time).total_seconds()

# Total audio duration estimate
for audio_array in audio_arrays:
    total_audio_duration_sec += len(audio_array) / 16000  # 16kHz

# Real Time Factor (RTF) calculation
real_time_factor = round(processing_seconds / total_audio_duration_sec, 4) if total_audio_duration_sec > 0 else None

# WER Summary
average_wer = round(np.mean(wer_scores), 4) if wer_scores else None

# Save Meta JSON
meta = {
    "processing_time_seconds": processing_seconds,
    "total_audio_duration_seconds": total_audio_duration_sec,
    "real_time_factor": real_time_factor,
    "number_of_files": len(audio_paths),
    "average_wer": average_wer,
    "batch_size": BATCH_SIZE,
    "long_form_generation_parameters": {
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
        "condition_on_prev_tokens": False,
        "return_timestamps": True,
        "max_length": 448,
        "no_repeat_ngram_size": 0,
    }
}

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nTranscriptions saved to {results_file}")
print(f"Meta information saved to {meta_file}")
print(f"Total processing time: {processing_seconds:.2f} seconds")
print(f"Total audio minutes: {total_audio_duration_sec/60:.2f} min")
print(f"Real-Time Factor (RTF): {real_time_factor}")
if average_wer is not None:
    print(f"Average WER (for files with gold transcriptions): {average_wer:.2%}")
else:
    print("No gold transcriptions found for WER calculation.")

