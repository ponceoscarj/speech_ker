from omegaconf import OmegaConf
from utils import DATA_DIR, WORK_DIR, ROOT
import os
import wget
from nemo.collections.asr.models import ClusteringDiarizer
from create_manifest import create_manifest
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

# diar_infer_general.yaml
# diar_infer_meeting.yaml
# diar_infer_telephonic.yaml
diarize_yml = "diar_infer_telephonic.yaml" 

MODEL_CONFIG = os.path.join(WORK_DIR,diarize_yml)
if not os.path.exists(MODEL_CONFIG):
    config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{diarize_yml}"
    MODEL_CONFIG = wget.download(config_url,WORK_DIR)

config = OmegaConf.load(MODEL_CONFIG)


output_manifest_name = 'diarize_manifest'
# if os.path.isfile(f'{WORK_DIR}/{output_manifest_name}.json'): 
#     None
# else:
create_manifest(data_dir= DATA_DIR, duration=None, work_dir= WORK_DIR, 
output_file_name=output_manifest_name, type='diarization', url_transcript_text='asr_work_dir/asr_outcomes/parakeet_tdt.txt')

## parameters for speaker diarization inference
pretrained_vad = 'vad_multilingual_marblenet'
pretrained_speaker_model = 'titanet_large'

config.num_workers = 0 # Workaround for multiprocessing hanging with ipython issue 
output_dir = os.path.join(ROOT, 'outputs')
config.diarizer.manifest_filepath = f'{WORK_DIR}/{output_manifest_name}.json'
config.diarizer.out_dir = output_dir #Directory to store intermediate files and prediction outputs

config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
config.diarizer.clustering.parameters.oracle_num_speakers=False

# config.diarizer.ignore_overlap = False

# Here, we use our in-house pretrained NeMo VAD model
config.diarizer.vad.model_path = pretrained_vad
# config.diarizer.vad.parameters.onset = 0.1
# config.diarizer.vad.parameters.offset = 0.1
# config.diarizer.vad.parameters.shift_length_in_sec = 0.05
# config.diarizer.vad.parameters.min_duration_off = 0

# config.diarizer.vad.parameters.pad_offset = -0.05
config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model 
config.diarizer.msdd_model.parameters.sigmoid_threshold = [1.0] # Evaluate with T=0.7 and T=1.0

# sd_model = ClusteringDiarizer(cfg=config)
# sd_model.diarize()

system_vad_msdd_model = NeuralDiarizer(cfg=config)
system_vad_msdd_model.diarize()