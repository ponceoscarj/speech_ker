# Benachmark of ASR and Diarization models

## ASR models
From hugging face - [open ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

## Diarization models
- NeMo (NVIDIA)
- Pyannote

Helpful articles:
- https://aclanthology.org/2024.fieldmatters-1.6.pdf


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