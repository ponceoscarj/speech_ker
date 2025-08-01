# adapted from https://github.com/huggingface/transformers/pull/27658
# paper https://cdn.openai.com/papers/whisper.pdf



'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python crisperwhisper.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/CrisperWhisper \
                --model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/CrisperWhisper \
                --output_dir sequential_whisper_largeV3 \
                --batch_size 8 \
                --timestamps none \
                --extensions .wav


Notes:
--extensions .wav . mp3 #can accept multiple
In --output_dir insert the correct model, be as specific as possible (e.g., canary-1b, canary-1b-flash, canary-180m)
'''
import argparse
import json
import os
import logging
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from jiwer import compute_measures
from itertools import islice         # ← ADD THIS
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

def read_gold_transcription(audio_path: Path) -> str | None:
    txt_path = audio_path.with_suffix('.txt')
    return txt_path.read_text().strip() if txt_path.exists() else None

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

def batch_iterator(iterator, batch_size: int):
    while batch := list(islice(iterator, batch_size)):
        yield batch

def process_batch(batch, processor, model, device, args, stats,batch_num):
    # Extract arrays & paths
    audio_arrays = [ex['audio']['array'] for ex in batch]
    paths = [Path(ex['audio']['path']).resolve() for ex in batch]

    # Inference
    start = time.time()

        # ADD THIS BLOCK
    start_time = time.time()
    start_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🟢 STARTING BATCH {batch_num} at {start_str}")
    print(f"🔢 Processing {len(batch)} files: {[p.name for p in paths]}")
    logging.info(f"Starting batch {batch_num} at {start_str} with {len(batch)} files")

    with torch.inference_mode(), torch.cuda.amp.autocast():
        inputs = processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
            truncation=False
        ).to(device)
        
        outputs = model.generate(
            **inputs,
            return_timestamps=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            condition_on_prev_tokens=False,
            return_legacy_cache=True
        )
    decode_time = time.time() - start

    # Decode & record
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    sampling_rate = processor.feature_extractor.sampling_rate
    batch_audio_dur = sum(len(arr) for arr in audio_arrays) / sampling_rate

    entries = []
    batch_wers = []

    for path, text in zip(paths, decoded):
        pred = text["text"] if isinstance(text, dict) else text
        entry = {"audio_file_path": str(path), "pred_text": pred}

        if args.gold_standard:
            gold = read_gold_transcription(path)
            if gold:
                entry["text"] = gold
                measures = compute_measures(gold, pred)
                entry.update({k: measures[k] for k in ('wer', 'mer', 'wil')})
                batch_wers.append(measures['wer'])                
                stats['total_wer'] += measures['wer']
                stats['count'] += 1
            else:
                entry["text"] = None
        entries.append(entry)

    # Update stats
    stats['rtf_list'].append(real_time_factor(decode_time, batch_audio_dur))
    stats['time'] += decode_time

    return entries, decode_time, [p.name for p in paths], batch_wers

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
    parser.add_argument("--batch_size", type=int, default=1,
                     help="Batch size for processing")
    parser.add_argument("--timestamps", choices=["word", "segment", "none"], default="word",
                     help="Type of timestamps to include")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"],
                     help="Audio file extensions to process")
    parser.add_argument("--gold_standard", action="store_true",default=True,
                       help="Enable WER calculation using gold standard transcriptions")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=Path(args.output_dir)/"transcription.log",
                        format="%(asctime)s %(levelname)s: %(message)s")

    # REMOVED: Log directory creation and log file setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    with tqdm(total=2, desc="Loading Model", unit="step") as bar:
        # Flash Attention 2 for 3-5x speedup
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        bar.update(1)
        model.generation_config.task = "transcribe"
        model.generation_config.language = "en"
        model.to(device)
        processor = AutoProcessor.from_pretrained(args.model)
        bar.update(1)
    
    model.to(device)

    exts = {e.lower() for e in args.extensions}
    audio_files = [p for p in Path(args.input_dir).rglob("*")
                   if p.suffix.lower() in exts]
    total_files = len(audio_files)
    if total_files == 0:
        logging.error("No audio files found. Exiting.")
        return

    dataset = load_dataset("audiofolder", data_dir=args.input_dir, streaming=True)["train"]
    dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
    data_iter = iter(dataset)


    stats = {"total_wer": 0.0, "count": 0, "rtf_list": [], "time": 0.0}
    all_results = []

    trans_bar = tqdm(total=total_files, 
                     desc="Transcribing", 
                     unit="file", 
                     colour="GREEN", 
                     position=0, 
                     leave=True, dynamic_ncols=True,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                     )        
    msg_bar = tqdm(
        total=0, 
        position=1, 
        leave=False, 
        bar_format="{desc}"
    )    
    # Process
    batch_num = 0
    for batch in batch_iterator(data_iter, args.batch_size):
        batch_num += 1
        try:
            entries, decode_time, names, batch_wers = process_batch(batch, processor, model, device, args, stats, batch_num)
            all_results.extend(entries)
            trans_bar.update(len(batch))

            avg_batch_wer = round(sum(batch_wers)/len(batch_wers), 4) if batch_wers else None
            # trans_bar.set_postfix({
            #     'batch_wer': avg_batch_wer if avg_batch_wer is not None else 'N/A',
            #     'batch_time_s': f"{decode_time:.2f}"
            # })

            # Update message bar below progress
            msg_bar.set_description_str(
                f"Batch {batch_num} | "
                f"Files: {names}\n"
                f"WERs: {batch_wers} (avg: {avg_batch_wer})\n"
                f"Time: {decode_time:.2f}s"
            )
            msg_bar.refresh()            

            for e in entries:
                wer_str = f"{e.get('wer'):.4f}" if 'wer' in e else 'N/A'
                logging.info(f"File: {e['audio_file_path']} → WER: {wer_str}")

        except Exception as e:
            logging.error(f"Batch error: {e}", exc_info=True)
        finally:
        # free memory
          torch.cuda.empty_cache()

    trans_bar.close()

    # Save outputs
    timestamp = args.output_filename or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.output_dir)/f"{timestamp}"
    with open(f"{out_base}.json", "w") as f:
        for entry in all_results:
            f.write(json.dumps(entry) + '\n')

    # Metadata
    avg_wer = stats['total_wer']/stats['count'] if stats['count'] else 0.0
    meta = {
        "processing_time": stats['time'],
        "real_time_factor": sum(stats['rtf_list'])/len(stats['rtf_list']) if stats['rtf_list'] else None,
        "average_wer": avg_wer,
        "batches_processed": len(stats['rtf_list']),
    }
    with open(f"{out_base}_meta.json", "w") as f:
        f.write(json.dumps(meta) + '\n')

    print(f"Overall average WER: {avg_wer:.4f}")
    print(f"Done! Results → {out_base}.json, {out_base}_meta.json")

if __name__ == "__main__":
    main()
