
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
from transformers.pipelines.automatic_speech_recognition import chunk_iter
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if "cuda" in device else torch.float32

# Load the compressed Whisper model
model = AutoModel.from_pretrained(
    "/home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/lite-whisper-large-v3", 
    trust_remote_code=True,
    torch_dtype=dtype
).to(device)

# Use the same processor as the original model
processor = AutoProcessor.from_pretrained("/home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/whisper-large-v3")

# Create ASR pipeline for chunk processing (no generation)
post_pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor)



# Load audio file
path = "/home/ext_ponceponte_oscar_mayo_edu/speech_ker/audio_files/audio_valid/afap024.wav"
audio, sr = librosa.load(path, sr=16000)

# Process audio in chunks using pipeline's preprocessing
# chunks = asr_pipe.preprocess(audio)
sr = processor.feature_extractor.sampling_rate

# align_to = model.config.inputs_to_logits_ratio
align_to = getattr(model.config, "inputs_to_logits_ratio", 1)
chunk_len    = int(round(30 * sr / align_to) * align_to)
stride_left  = int(round( 5 * sr / align_to) * align_to)
stride_right = int(round( 5 * sr / align_to) * align_to)

chunks = list(chunk_iter(
    audio, 
    processor.feature_extractor, 
    chunk_len, 
    stride_left, 
    stride_right,
    dtype=torch.float32
))




model_outputs = []
for chunk in tqdm(chunks, desc="Processing chunks"):
    print('\n', chunk)
    # Extract and prepare inputs
    outputs = model.generate(
                    input_features=chunk["input_features"].to(device).to(dtype),
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"),
                    max_new_tokens=444)
    print('outputs', outputs)
    tokens_cpu = outputs.cpu().detach()

    model_outputs.append({"tokens": tokens_cpu, "stride": chunk["stride"]})

final = post_pipe.postprocess(model_outputs)

# Combine chunks with overlap handling
print(final["text"])