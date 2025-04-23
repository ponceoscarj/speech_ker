
'''
Modification of https://github.com/nyrahealth/CrisperWhisper/blob/main/transcribe.py 

Example:
python crisperwhisper.py --input_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files \
                --output_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/output/CrisperWhisper \
                --main_model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/CrisperWhisper \
                --processor_model /Users/oscarponce/Documents/PythonProjects/speech_ker/asr/models/whisper-large-v3 \
                --chunk_length 30 \
                --batch_size 1 \
                --extensions .wav


Notes:
--extensions .wav . mp3 #can accept multiple
In --output_dir insert the correct model, be as specific as possible (e.g., canary-1b, canary-1b-flash, canary-180m)
'''
import librosa 
import torch
from transformers import AutoProcessor, AutoModel, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if "cuda" in device else torch.float32

# Load the compressed Whisper model
model = AutoModel.from_pretrained(
    "efficient-speech/lite-whisper-large-v3-turbo", 
    trust_remote_code=True,
    torch_dtype=dtype
).to(device)

# Use the same processor as the original model
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# Create ASR pipeline for chunk processing (no generation)
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,        # 30-second chunks
    # stride_length_s=[6, 4],   # [chunk_stride, label_stride]
    device=device,
    torch_dtype=dtype,
)

# Load audio file
path = "path/to/audio.wav"
audio, sr = librosa.load(path, sr=16000)

# Process audio in chunks using pipeline's preprocessing
chunks = asr_pipe.preprocess(audio)

transcriptions = []
for chunk in tqdm(chunks, desc="Processing chunks"):
    # Extract and prepare inputs
    inputs = {
        "input_features": chunk["input_features"].to(device).to(dtype),
        "decoder_input_ids": torch.tensor(
            [[model.config.decoder_start_token_id]],
            device=device
        )
    }
    
    # Generate with forced decoder prompts
    predicted_ids = model.generate(
        **inputs,
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            language="en", 
            task="transcribe"
        ),
        max_new_tokens=448,
        num_beams=1,
        do_sample=False
    )
    
    # Decode and clean text
    text = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0].strip()
    
    transcriptions.append(text)

# Combine chunks with overlap handling
full_transcription = " ".join(transcriptions)
print("Final Transcription:")
print(full_transcription)