#
# Adapted from microsoft/PhiCoocBook https://github.com/microsoft/PhiCookBook 
# example url: https://github.com/microsoft/PhiCookBook/tree/main/md/02.Application/05.Audio/Phi4/Transciption
# 

'''
Usage example

'''


import torch
import soundfile as sf
import numpy as np
from collections import deque
from transformers import AutoModelForCausalLM, AutoProcessor
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.pipeline import Pipeline
from pyannote.core import Segment
import math

# Configuration
MODEL_PATH = 'Your Phi-4-multimodal location'
CHUNK_DURATION = 30          # Target chunk size in seconds
CONTEXT_WORDS = 30           # Number of context words to maintain
OVERLAP_SECONDS = 1.5        # Audio overlap in seconds
PYANNOTE_TOKEN = "your_huggingface_token"

PROMPT_TEMPLATE = """<|user|><|audio_1|>
Generate detailed transcription continuing from:
{text_context}
<|end|><|assistant|>
"""

def get_device() -> torch.device:
    """Return the appropriate device ('cuda' if available, else 'cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models() -> tuple:
    """
    Load the processor, generation model, and voice activity detection pipeline.
    - Uses device mapping and memory optimizations.
    """
    device = get_device()
    
    # Load the processor once (for consistent tokenization and feature extraction)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Load the model with optimized parameters:
    # - Use bfloat16 on CUDA for lower memory footprint.
    # - Use flash_attention if available.
    # - Use device_map="auto" to ease device placement.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    attn_implementation = "flash_attention_2" if torch.cuda.is_available() else None
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map="auto"
    )
    
    # Load a single instance of the Pyannote VAD pipeline with hardware awareness.
    segmentation_model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=PYANNOTE_TOKEN
    ).to(device)
    
    vad_pipeline = VoiceActivityDetection(segmentation=segmentation_model)
    vad_pipeline.to(device)

    HYPER_PARAMETERS = {
        "onset": 0.5, 
        "offset": 0.5,
        "min_duration_on": 0.1,
        "min_duration_off": 0.1,
    }
    vad_pipeline.instantiate(HYPER_PARAMETERS)    
    
    return processor, model, vad_pipeline, device

def pyannote_chunking(vad_pipeline: Pipeline, audio_path: str, chunk_duration: float) -> list:
    """
    Run voice activity detection on the audio file and split speech segments
    into smaller chunks of up to `chunk_duration` seconds.
    """
    vad_results = vad_pipeline(audio_path)
    speech_segments = []
    current_segment = None
    
    # Merge consecutive speech segments
    for segment, _, label in vad_results.itertracks(yield_label=True):
        if label == "speech":
            if current_segment is None:
                current_segment = segment
            else:
                # Extend the current segment—take the maximum end time.
                current_segment = Segment(current_segment.start, segment.end)
        else:
            if current_segment and (current_segment.end - current_segment.start) > 0.1:
                speech_segments.append(current_segment)
            current_segment = None
    
    # Don't forget the last segment if it's speech
    if current_segment and (current_segment.end - current_segment.start) > 0.1:
        speech_segments.append(current_segment)
    
    # Split each speech segment into sub-chunks of length <= chunk_duration seconds.
    chunks = []
    for segment in speech_segments:
        segment_duration = segment.end - segment.start
        num_subchunks = math.ceil(segment_duration / chunk_duration)
        for i in range(num_subchunks):
            sub_start = segment.start + i * chunk_duration
            sub_end = min(segment.end, sub_start + chunk_duration)
            chunks.append(Segment(sub_start, sub_end))
    
    return chunks

@torch.inference_mode()  # More efficient than torch.no_grad()
def process_audio(audio_path: str, chunk_duration: float = CHUNK_DURATION) -> str:
    """
    Process the entire audio file:
    - Load the audio once with memory mapping.
    - Convert multi-channel audio to mono.
    - Use VAD to chunk the audio.
    - Generate transcriptions for each chunk and maintain context.
    """
    # Load models and device info.
    processor, model, vad_pipeline, device = load_models()
    
    # Read audio with memory mapping; force 2D array to handle even mono files uniformly.
    audio, sr = sf.read(audio_path, always_2d=True)
    # Average across channels to get mono audio.
    audio = np.mean(audio, axis=1)
    
    # Optimized tensor creation and device transfer
    audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=device)
    if device.type == 'cuda':
        audio_tensor = audio_tensor.pin_memory().to(device, non_blocking=True)
    else:
        audio_tensor = audio_tensor.to(device)

    # Use the provided VAD pipeline to obtain speech chunks.
    chunks = pyannote_chunking(vad_pipeline, audio_path, chunk_duration)
    
    # Context buffers for text and audio.
    text_context = deque(maxlen=CONTEXT_WORDS)
    full_transcript = []

    # Calculate buffer length once
    audio_buffer = torch.empty(0, dtype=torch.float32, device=device)
    buffer_samples = int(OVERLAP_SECONDS * sr)

    prev_chunk_end = 0  # Track previous chunk's end time

    # Process each detected speech chunk.
    for chunk in chunks:
        if prev_chunk_end != 0 and (chunk.start - prev_chunk_end) > OVERLAP_SECONDS:
            audio_buffer = torch.empty(0, dtype=torch.float32, device=device)        
        # Get current chunk audio
        start_idx = int(chunk.start * sr)
        end_idx = int(chunk.end * sr)
        current_audio = audio_tensor[start_idx:end_idx]


        # Combine with previous buffer
        chunk_audio = torch.cat([audio_buffer, current_audio])
        
        # Update buffer for next iteration
        audio_buffer = current_audio[-buffer_samples:] if current_audio.size(0) >= buffer_samples else current_audio

        context_text = " ".join(text_context)
        inputs = processor(
            text=PROMPT_TEMPLATE.format(text_context=context_text),
            audios=[chunk_audio.cpu().numpy()],
            return_tensors="pt",
            sampling_rate=sr,
            padding='longest',
            truncation=True
        ).to(device)
        
        # Generate transcription with tuned parameters.
        gen_config = {
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            # "do_sample": True,
            "use_cache": True,
            "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        }
        
        outputs = model.generate(**inputs, **gen_config)
        
        # Slice off the input tokens and decode only the newly generated ones.
        input_length = inputs['input_ids'].size(1)
        transcript = processor.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()
        
        full_transcript.append(transcript)
        # Extend text context with the last CONTEXT_WORDS from this transcript.
        text_context.extend(transcript.split()[-CONTEXT_WORDS:])
        prev_chunk_end = chunk.end

    return " ".join(full_transcript)

if __name__ == "__main__":
    # Set cudnn benchmark for improved performance on fixed-size inputs.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True        
        

    transcription = process_audio("input.wav")
    print("Final Transcription:\n", transcription)
