
'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python ibm-granite.py \
                --input_dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/audio_files/valid_audio \
                --output_dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/output/ibm-granite \
                --model /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models_outputs \
                --output_filename ibm-granite \
                --chunk-lengths 30 \
                --batch-sizes 1 \
                --sleep-time 5 \
                --extensions .wav \
                --system_prompt /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/ibm-granite/system-prompt_ibm-granite.txt \
                --user_prompt /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/ibm-granite/user-prompt_ibm-granite.txt

'''
import argparse
import json
import os
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import warnings
from tqdm import tqdm
import time
import soundfile as sf

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

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

def txt_prompts(file_url):
    with open(file_url, 'r') as f:
       prompt = f.read().strip()
    return prompt

def chunk_audio(audio_array, sr, chunk_sec):
    chunk_size = int(sr * chunk_sec)
    return [audio_array[i:i + chunk_size] for i in range(0, len(audio_array), chunk_size)]

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
    parser.add_argument("--gold_standard", action="store_true", default=True,
                        help="Enable WER calculation using gold standard transcriptions")
    parser.add_argument("--sleep-time", type=int, default=0,
                        help="Optional sleep time between batches (not used currently)")
    parser.add_argument("--system_prompt", type=str,
                        help="Prompt instruction for the system")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"],
                     help="Audio file extensions to process")    
    parser.add_argument("--user_prompt", type=str, 
                        help="Prompt instruction for the user")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    # ── Load model & processor ─────────────────────────────────────────────
    with tqdm(total=2, desc="Loading Model") as bar:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        bar.update(1)
        processor = AutoProcessor.from_pretrained(args.model)

        tokenizer = processor.tokenizer
      
       # ── HERE ── get the special decoder prompt for Whisper
        # forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        system_prompt = txt_prompts(args.system_prompt)
        user_prompt = txt_prompts(args.user_prompt)
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        bar.update(1)

  # ── Transcription Loop ────────────────────────────────────────────────
    files = []
    for ext in args.extensions:
        files.extend(Path(args.input_dir).rglob(f"*{ext}"))
    files = sorted(files)
    total_files = len(files)
    
    main_bar = tqdm(total=total_files, desc="Transcribing", unit="file")
    # wer_bar = tqdm(total=total_files, desc="WER_Calc", leave=False)
    results, batch_rtfs = [], []
    # total_rwer, total_nwer = 0.0, 0.0
    # total_insertions = 0
    # total_deletions = 0
    # total_substitutions = 0
    # wer_count = 0
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
                model_inputs = processor(prompt,
                    to_process,
                    return_tensors="pt",
                    device=device,
                    # padding=True
                ).to(device)

                # print("Model Inputs:")
                # for key, value in model_inputs.items():
                #     if isinstance(value, torch.Tensor):
                #         print(f"{key}: shape={value.shape}, device={value.device}")                
    
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    model_outputs = model.generate(**model_inputs, 
                                              max_new_tokens=200,
                                              do_sample=False,
                                              num_beams=1)
                    num_input_tokens = model_inputs["input_ids"].shape[-1]

                    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
                    outputs = tokenizer.batch_decode(
                        new_tokens, add_special_tokens=False, skip_special_tokens=True
                    )  

                    # outputs = processor.batch_decode(pred_ids, skip_special_tokens=True)
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
                  
            results.append(entry)
            main_bar.update(1)
    
        if args.sleep_time > 0:
            time.sleep(args.sleep_time)
    
    main_bar.close()
    total_time = time.time() - start_all
    
      
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
            "total_audio_duration_s": total_audio_duration,
            "total_n_files": total_files,
            "real_time_factor": rtf_all,
            "batch_rtfs": batch_rtfs
        }, f, indent=2)
    
    print(f"\nResults → {results_file}")
    print(f"Metadata → {meta_file}")
    print(f"Total Time: {total_time:.2f}s  Audio: {total_audio_duration / 60:.2f}m  RTF: {rtf_all:.4f}")

if __name__ == "__main__":
    main()
