from transformers import pipeline
from pydub import AudioSegment
import os

original_file_path = "/Users/yuqiwu/Documents/PythonProjects/ASR_chunks_2025/audio_files/toy2.wav"
audio = AudioSegment.from_file(original_file_path)

chunk_length = len(audio) // 10
overlap = chunk_length // 10  # 10% overlap

# Get the base filename without the extension
base_filename = os.path.splitext(os.path.basename(original_file_path))[0]

# Split the audio and export each chunk with the new naming convention
for i in range(10):
    start = i * chunk_length
    if i != 0:  # Start earlier for overlap, except for the first chunk
        start -= overlap
    end = start + chunk_length + overlap  # Extend the end for overlap
    if i == 9:  # Ensure the last chunk includes any leftover portion
        end = len(audio)
    chunk = audio[start:end]
    chunk_filename = f"/Users/yuqiwu/Documents/PythonProjects/ASR_chunks_2025/audio_files/chunked_audio/{base_filename}_chunk{i+1}.wav"  # Format filename
    chunk.export(chunk_filename, format="wav")