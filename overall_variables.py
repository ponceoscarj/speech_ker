import os
import numpy as np
import nemo
import pprint
from omegaconf import OmegaConf
import shutil
import wget


ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT,'audio_files')
WORK_DIR = os.path.join(ROOT, 'asr_work_dir')

# Check URLs
# print(ROOT)
# print(DATA_DIR)
# print(WORK_DIR)

# # AUDIO_FILENAME = os.path.join(data_dir,'toy1.wav')

# signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=None)


# DOMAIN_TYPE = "meeting" # Can be meeting or telephonic based on domain type of the audio file
# CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
# CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

# if not os.path.exists(os.path.join(WORK_DIR,CONFIG_FILE_NAME)):
#     CONFIG = wget.download(CONFIG_URL, WORK_DIR)
# else:
#     CONFIG = os.path.join(WORK_DIR,CONFIG_FILE_NAME)

# cfg = OmegaConf.load(CONFIG)

