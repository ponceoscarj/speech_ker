import os

# AUDIO_PATH = '/Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files/toy2.wav'

# print('NAME\n')
# print(os.path.basename(AUDIO_PATH))
from nemo.collections.asr.models import EncDecMultiTaskModel
# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash')