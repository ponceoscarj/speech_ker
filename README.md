# Benachmark of ASR and Diarization models

## ASR models
All files for asr inference are in the `asr` folder with instructions on how to run the models. 


## Diarization models
All files for diarization inference are in the `diairization` folder with instructions on how to run the models. 


## Additional files

- align.py: File to create the manifest for forced alignment. 
- aligh.sh: Bash file to force the timeframes to each word or segment.
- join_diarize_asr.py: Joins all outputs into a single file. 
- keepalive.sh: allows the VPN to keep working and avoid inactivity issues. By default the Mayo Clinic VPN disconnects any user who is inactive for 1h. This file opens www.google.com on Microsoft Edge and closses all windows from Microsoft Edge. Do all your work on Chrome.
```bash
bash keepalive.sh # do this on git bash
```

Split an audio with:
- split_audio.py


## Patient-doctor conversations - toy examples

- OSCE conversations: https://springernature.figshare.com/articles/dataset/Collection_of_simulated_medical_exams/16550013?backTo=%2Fcollections%2FA_dataset_of_simulated_patient-physician_medical_interviews_with_a_focus_on_respiratory_cases%2F5545842&file=30598530

- Primock57: https://github.com/babylonhealth/primock57
