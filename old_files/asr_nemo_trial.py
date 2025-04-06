import nemo.collections.asr as nemo_asr
import os
from create_manifest import create_manifest
from old_files.utils import DATA_DIR, WORK_DIR, ROOT
from omegaconf import OmegaConf, open_dict
import nemo.collections.nlp as nemo_nlp
import re

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")

import os
import soundfile as sf
import nemo.collections.asr as nemo_asr

def transcribe_audio(audio_file, model):
    audio, sr = sf.read(audio_file)
    # Assuming the RNN-T model can handle numpy arrays directly; otherwise, you might need preprocessing
    transcription = model.transcribe(paths2audio_files=[audio_file], batch_size=1)
    return transcription[0]  # Assuming single file processed, returns the transcript string

def write_ctm(transcript, audio_file, output_path):
    # Placeholder for actual timestamped words
    # This part will need adjustment based on how you obtain word timings (not directly supported by all models)
    words = transcript.split()
    fake_start_time = 0.0
    duration = 0.5  # Placeholder duration

    # Prepare CTM content and save in a specific path
    ctm_filename = os.path.join(output_path, os.path.basename(audio_file) + ".ctm")
    with open(ctm_filename, "w") as f:
        for word in words:
            f.write(f"{os.path.basename(audio_file)} 1 {fake_start_time:.2f} {duration:.2f} {word}\n")
            fake_start_time += duration  # Increment start time by duration as a placeholder

def process_directory(input_folder, output_folder, model):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each WAV file in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".wav"):
            audio_file = os.path.join(input_folder, filename)
            transcript = transcribe_audio(audio_file, model)
            write_ctm(transcript, audio_file, output_folder)


# Specify your input and output directories
input_folder = '/Users/yuqiwu/Documents/PythonProjects/ASR_chunks_2025/audio_files/chunked_audio'
output_folder = '/Users/yuqiwu/Documents/PythonProjects/ASR_chunks_2025/asr_work_dir/nfa_output/ctm'

# Process the directory
process_directory(input_folder, output_folder, asr_model)

# for file in sorted(os.listdir(DATA_DIR), key=lambda x: int(''.join(filter(str.isdigit, x)))):
#     filename, ext = os.path.splitext(file)
#     print(filename)
#     create_manifest(data_dir= DATA_DIR, duration=None, work_dir= WORK_DIR, output_file_name=f'output_manifest_{filename}', type="asr")
#
#
#
#
# # # print(os.path.join(WORK_DIR, f'{output_manifest_name}.json'))
#
#
# #obtain all audio files to be transcribed
# lst_audio = []
# for file in os.listdir(f'{DATA_DIR}'):
#     lst_audio.append(os.path.join(DATA_DIR, file))
#
# print(lst_audio)
#
# # # predicted_text = asr_model.transcribe(
# # #     audio=f"ask_work_dir/{output_manifest_name}.json",
# # #     batch_size=16,  # batch size to run the inference with
# # # )
#
#
# # # load model for punctuation
# punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(model_name='punctuation_en_bert')
#
#
# audio_files = sorted(
#     [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.wav')],
#     key=lambda x: int(''.join(filter(str.isdigit, x)))
# )
#
#
# predicted_text = asr_model.transcribe(
#     paths2audio_files=audio_files,
#     batch_size=16)
#
# # print(predicted_text)
# # print(predicted_text[0])
#
# # Add punctuation + save
# with open(f'{WORK_DIR}/asr_outcomes/parakeet_tdt_trial.txt', 'w') as f:
#     for i in predicted_text[0]:
#         # i = punctuation.add_punctuation_capitalization([i])[0]
#         f.write(i+'\n')
#
#
# with open('asr_work_dir/asr_outcomes/parakeet_tdt_trial.txt', "r") as f:
#     text = f.readlines()
#     # print(text)
#     create_punct = []
#     # align_file = []
#     for i in text:
#         print(i.rstrip())
#         i = i.rstrip()
#         i = punctuation.add_punctuation_capitalization([i])
#         create_punct.append(i)
#         # align_i = i.replace(".", "|")
#         # align_file.append(align_i)
#
#     with open('asr_work_dir/asr_outcomes/parakeet_tdt_punct_trial.txt', "w") as f:
#         for i in create_punct:
#             f.write(i[0]+'\n')
#
#     with open('asr_work_dir/asr_outcomes/parakeet_tdt_align_trial.txt', "w") as f:
#         for i in create_punct:
#             new_line = re.sub(r"\?|\!|\.", "|", i[0])
#             f.write(new_line+'\n')
#
#
# # PUNCTUATION + CAPITALIZATION
# # punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(model_name='punctuation_en_bert')
#
#
#
#
#
#
#
#
# # if hypotheses form a tuple (from RNNT), extract just "best" hypotheses
# # if type(hypotheses) == tuple and len(hypotheses) == 2:
# #     hypotheses = hypotheses[0]
#
# # timestamp_dict = hypotheses[0].timestep # extract timesteps from hypothesis of first (and only) audio file
# # print("Hypothesis contains following timestep information :", list(timestamp_dict.keys()))
#
# # print(predicted_text)
# # print(type(predicted_text))
#
# # for i in predicted_text[0]:
# #     print(i)





