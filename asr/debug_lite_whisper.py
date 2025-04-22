# debug_lite_whisper.py
import librosa
import torch
from transformers import AutoModel, AutoProcessor

# 1) Point to your model and test audio:
MODEL_PATH = "/home/ext_alzahidy_misk_mayo_edu/speech_ker/asr/models/lite_whisper_large_v3"
AUDIO_PATH = "/home/ext_alzahidy_misk_mayo_edu/speech_ker/audio_files/audio_valid/afap024.wav"

# 2) Set up device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device.startswith("cuda") else torch.float32

# 3) Load model & processor
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto"
).to(device)
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# 4) Load the audio
audio, _ = librosa.load(AUDIO_PATH, sr=16000)
print(f"Loaded audio: {len(audio)} samples (~{len(audio)/16000:.2f}s)")

# 5) Preprocess
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
input_feats = inputs.input_features.to(device).to(dtype)
print("Input features shape:", input_feats.shape)

# 6) Generate
with torch.inference_mode():
    pred_ids = model.generate(input_feats)
print("Generated token IDs:", pred_ids)

# 7) Decode
text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
print("Transcription:", repr(text))
