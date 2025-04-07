# Benachmark of Diarization models


## Diarization models
- NeMo (NVIDIA)


- Pyannote
Install pyannote 
```bash
pip install pyannote.audio
```

Steps
1. Create an account on huggingface and generate a token. https://huggingface.co/
2. Grant access on https://huggingface.co/pyannote/speaker-diarization-3.1
3. Grant access on https://huggingface.co/pyannote/segmentation-3.0


Helpful articles:
- Diarization: https://aclanthology.org/2024.fieldmatters-1.6.pdf


## Files for Inference

Required files for ASR + diarization:
- asr_nemo.py: Transform speech to text. Requires a manifest.
- diarization.py: Diarize speech. Output is the time frame per speaker. No text is needed in this step. Requires a manifest + .yaml file. 
- align.py: File to create the manifest for forced alignment. 
- split.sh: Bash file to force the timeframes to each word or segment.
- join_diarize_asr.py: Joins all outputs into a single file. 

Split an audio with:
- split_audio.py

Working with manifest
- create_manifest.py: All models require this file to create a manifest


## Patient-doctor conversations - toy examples

- OSCE conversations: https://springernature.figshare.com/articles/dataset/Collection_of_simulated_medical_exams/16550013?backTo=%2Fcollections%2FA_dataset_of_simulated_patient-physician_medical_interviews_with_a_focus_on_respiratory_cases%2F5545842&file=30598530

- Primock57: https://github.com/babylonhealth/primock57
