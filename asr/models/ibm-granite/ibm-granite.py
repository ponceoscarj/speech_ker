
'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python ibm-granite.py \
                --input_dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/audio_files/valid_audio \
                --output_dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/output/ibm-granite \
                --model /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models_outputs \
                --output_filename ibm-granite \
                --chunk_lengths 30 \
                --batch_sizes 1 \
                --max_new_tokens 200 \
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

def real_time_factor(processing_time, audio_length, decimals=4):
    return round(processing_time / audio_length, decimals) if audio_length > 0 else None

def read_gold_transcription(audio_path):
    txt_path = Path(audio_path).with_suffix('.txt')
    return txt_path.read_text().strip() if txt_path.exists() else None

def txt_prompts(file_url):
    return Path(file_url).read_text().strip()

def main():
    parser = argparse.ArgumentParser("Batch chunked ASR with timing metrics")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="transcripts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_filename", default="")
    parser.add_argument("--chunk_lengths", type=int, default=30,
                        help="length of each chunk in seconds")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="maximum number of tokens in the output")
    parser.add_argument("--batch_sizes", type=int, default=4,
                        help="number of chunks per batch")
    parser.add_argument("--extensions", nargs='+', default=[".wav"])
    parser.add_argument("--system_prompt", required=True)
    parser.add_argument("--user_prompt", required=True)
    parser.add_argument("--gold_standard", action='store_true')
    parser.add_argument("--sleep-time", type=int, default=0,
                        help="Optional sleep time between batches (not used currently)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device.startswith('cuda') else torch.float32

    # load model & processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        device_map='auto', low_cpu_mem_usage=True
    ).to('cuda')
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    # prepare prompts
    chat = [
        {"role": "system", "content": txt_prompts(args.system_prompt)},
        {"role": "user", "content": txt_prompts(args.user_prompt)},
    ]
    prompt_ids = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # collect all chunks across files
    chunk_infos = []  # list of dicts: {path, array, duration}
    for ext in args.extensions:
        for path in sorted(Path(args.input_dir).rglob(f"*{ext}")):
            audio, sr = sf.read(path)
            chunk_size = sr * args.chunk_lengths
            duration = len(audio) / sr
            # split into chunks
            for i in range(0, len(audio), chunk_size):
                segment = audio[i:i+chunk_size]
                seg_dur = len(segment) / sr
                if seg_dur > 0:
                    chunk_infos.append({
                        'path': path,
                        'array': segment,
                        'duration': seg_dur
                    })
    total_audio = sum(ci['duration'] for ci in chunk_infos)
    total_files = len({ci['path'] for ci in chunk_infos})

    results_per_file = {str(p): [] for p in set(ci['path'] for ci in chunk_infos)}
    batch_rtfs = []

    # batch inference
    start_all = time.time()
    for i in tqdm(range(0, len(chunk_infos), args.batch_sizes), desc='Batches'):
        batch = chunk_infos[i:i+args.batch_sizes]
        inputs = [ci['array'] for ci in batch]
        durations = [ci['duration'] for ci in batch]
        t0 = time.time()
        inputs_proc = processor(
            [prompt_ids]*len(inputs),
            inputs,
            return_tensors='pt',
            padding=True
        ).to(device)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.generate(**inputs_proc, max_new_tokens=args.max_new_tokens)
        # split off prompt tokens
        n_prompt = inputs_proc['input_ids'].shape[-1]
        texts = tokenizer.batch_decode(
            outputs[:, n_prompt:], skip_special_tokens=True
        )
        t1 = time.time()

        # accumulate per-chunk RTF
        batch_time = t1 - t0
        for ci, txt in zip(batch, texts):
            results_per_file[str(ci['path'])].append(txt.strip())
        rtf = real_time_factor(batch_time, sum(durations))
        batch_rtfs.append(rtf)
        print(f"Batch {i//args.batch_sizes+1:04d}: chunks={len(batch)} RTF={rtf}")
        if args.sleep_time > 0:
            time.sleep(args.sleep_time)        

    total_time = time.time() - start_all
    rtf_all = real_time_factor(total_time, total_audio)

    # write results
    out_base = args.output_filename or f"results_{datetime.now().isoformat()}"
    res_file = os.path.join(args.output_dir, f"{out_base}.json")
    meta_file = os.path.join(args.output_dir, f"{out_base}_meta.json")
    with open(res_file, 'w') as f:
        for path, texts in results_per_file.items():
            entry = {'audio_file_path': path, 'pred_text': ' '.join(texts)}
            if args.gold_standard:
                entry['text'] = read_gold_transcription(path) or 'N/A'
            f.write(json.dumps(entry) + '\n')
    with open(meta_file, 'w') as f:
        json.dump({
            'processing_time_s': total_time,
            'total_audio_duration_s': total_audio,
            'total_n_files': total_files,
            'real_time_factor': rtf_all,
            'batch_rtfs': batch_rtfs
        }, f, indent=2)

    print(f"Results → {res_file}")
    print(f"Metadata → {meta_file}")
    print(f"Total Time: {total_time:.2f}s  Audio: {total_audio/60:.2f}m  RTF: {rtf_all:.4f}")

if __name__ == "__main__":
    main()
