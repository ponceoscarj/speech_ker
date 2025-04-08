import argparse
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python crisperwhisper.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/CrisperWhisper \
                --model openai/whisper-large-v3 \
                --chunk_length 30 \
                --batch_size 1 \
                --timestamps none \
                --extensions .wav

--extensions .wav . mp3 #can accept multiple
'''
def transcribe_audio(file_path, model_id, chunk_length, batch_size, return_timestamps):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "nyrahealth/CrisperWhisper"  # You can change this to a different model if needed

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length, # 30
        batch_size=batch_size, # 1
        return_timestamps=return_timestamps,
        torch_dtype=torch_dtype,
        device=device
    )

    return pipe(file_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="transcripts",
                        help="Output directory for transcriptions")
    parser.add_argument("--model", type=str, default="nyrahealth/CrisperWhisper",
                        help="ASR model identifier from Hugging Face Hub")
    parser.add_argument("--chunk_length", type=int, default=30,
                      help="Length of audio chunks in seconds")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for processing")
    parser.add_argument("--timestamps", choices=["word", "segment", "none"], default="word",
                      help="Type of timestamps to include")
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".mp3", ".flac"],
                      help="Audio file extensions to process")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = [
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in args.extensions
    ]

    if not audio_files:
        print(f"No audio files found with extensions {args.extensions} in {args.input_dir}")
        sys.exit(1)

    for audio_file in audio_files:
        file_path = os.path.join(args.input_dir, audio_file)
        print(f"\nProcessing {audio_file}...")
        
        try:
            result = transcribe_audio(
                file_path=file_path,
                model_id=args.model,
                chunk_length=args.chunk_length,
                batch_size=args.batch_size,
                return_timestamps=args.timestamps if args.timestamps != "none" else False
            )
            
            # Save results
            output_file = os.path.join(args.output_dir, f"{os.path.splitext(audio_file)[0]}_transcript.txt")
            with open(output_file, "w") as f:
                f.write(result["text"])
                if args.timestamps != "none":
                    f.write("\n\nTimestamps:\n")
                    f.write("\n".join([str(ts) for ts in result.get("chunks", [])]))
            
            print(f"Successfully processed {audio_file}")
            print(f"Transcript saved to {output_file}")

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
