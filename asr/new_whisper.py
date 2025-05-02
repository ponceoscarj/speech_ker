# code coming from https://github.com/huggingface/transformers/pull/27658
# paper https://cdn.openai.com/papers/whisper.pdf

from transformers import WhisperForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio
import torch
import numpy as np

processor = AutoProcessor.from_pretrained("/home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("/home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/whisper-large-v3", torch_dtype=torch.float16)
model.to("cuda")

# rertieve 8 long audio sequences
print('loading dataset')
ds = load_dataset("/home/ext_ponceponte_oscar_mayo_edu/speech_ker/audio_files/audio_files")["train"]
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
print('Taking batch size of 8')
# ds = ds[:8] # take batch size of 8
# print(ds)

print('printing audio list')

for i in ds["audio"]:
    print(i)

raw_audio = [x["array"].astype(np.float32) for x in ds["audio"]]

# process input, make sure to pass `padding='longest'` and `return_attention_mask=True`
inputs = processor(raw_audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
inputs = inputs.to("cuda", torch.float16)

# activate `temperature_fallback` and repetition detection filters and condition on prev text
result = model.generate(**inputs, condition_on_prev_tokens=False, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), logprob_threshold=-1.0, compression_ratio_threshold=1.35, return_timestamps=True)

decoded = processor.batch_decode(result, skip_special_tokens=True)
print(decoded)