# Benchmark of ASR models

The list of models to be evaluated are available on [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

The architecture characteristics of the models are stated at the end of this `README` file. 

## Requirements
### 1. Create conda environment 
```bash
conda create --name speech_ker python==3.10.12 # more stable
conda activate speech_ker
```

### 2. Install torch & transformers
Your torch must be compatible with your CUDA version. Check [here](https://pytorch.org/get-started/locally/).
```bash
# Check cuda version
nvidia-smi
# The CUDA version of our cloud is 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install NeMo 
#### Dependencies
```bash
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
```

#### NeMo
```bash
pip install "nemo_toolkit[all]"===999 #version 999 does not exist but it will trigger nemo_toolkit to show you all the version, choose the latest.
# pip install "nemo_toolkit[all]"===2.3.0rc2 #this is the latest version as of April 11th, 2025 - 
pip install "nemo_toolkit[all]"===2.4.0rc2
```

### 5. Install Huggingface Hub
Needed to download all ASR models. 
```bash
pip install -U "huggingface_hub[cli]"
```

### 6. Install other dependencies
```bash
pip install jiwer
pip install librosa
pip install --upgrade transformers 
```

### 7. Install FlashAttention [OPTIONAL]
FlashAttention can speed up both training and inference. [Link](https://github.com/Dao-AILab/flash-attention) for a detailed explanation. 

#### Requirements
- CUDA 12.0 and above (Check version with `nvidia-smi`)
- CUDA toolkit (Chech with `nvcc --version`)
```bash
pip install packaging
pip install ninja
```
- Check `ninja` is installed correctl with `ninja --version` then `echo $?`. This should return exit code `0`. If not, check flash attention link to debug.

#### Flash Attention
**Option 1:** Should work for most environments:
```bash
pip install flash-attn --no-build-isolation
```
**Optiona 2:** if your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might run too manu prallel compilation jobs that could exhaust the amount of RAM. To limit the number of parallale compilation jobs, you can set the environment variable with `MAX_JOBS` before installation
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```


## Preparing dataset
The data required for running the ASR benchamrk are: 1) audio files saved in `.wav` format and 2) gold-standard transcriptions (human annotation) saved in the `.txt` format. Files corresponding to the same  should have the same name if they correspond to the same conversation and saved under a unique validation or testing folder. 
```
audio_files
│   
└───valid_files
│   │   afap024.wav
│   │   afap024.txt
│   └── ...
└───test_files
    │   afpp294.wav
    │   afpp294.txt
    └── ...
```

## Validation and Testing
Your dataset should be randomly split into 20% and 80% for validation and testing respectively. Validation will consist of hyperparameter tuning. Because none of the ASR models was trained to transcribe long audios, they require chunking with or without stride ([explanation](https://huggingface.co/blog/asr-chunking)).  Parameters like this need to be tuned and the list of parameters to be tuned are available within each model's folder. The parameters to be tuned are available within each folder.


# CrisperWhisper and Distil-Whisper-large-v3.5

## Instructions
1. Create environment with Python 3.10 (more stable) - use your nemo conda environment if you have it
 
2. Clone repository into your `asr/models` folder **(only for CrisperWhisper)**:
```bash
git clone https://github.com/nyrahealth/CrisperWhisper.git
cd CrisperWhisper
```

3. Install the dependencies
 ```bash
 pip install -r ./CrisperWhisper/requirements.txt  # if you’re using CrisperWhisper
 pip install jiwer # for WER calculation
 ```


5. Install the models
- Login into huggingface and introduce your token. 
```bash
huggingface-cli login
```

- Download model files to a specific folder:
  
# CrisperWhisper
- Accept the license of the model [CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper)
```bash
huggingface-cli download nyrahealth/CrisperWhisper --local-dir [SAVE_DIR]
```

# Distil-Whisper
```bash
huggingface-cli download distil-whisper/distil-large-v3.5 --local-dir [SAVE_DIR]
```

*`[SAVE_DIR]` is your target directory. Inside asr/models. Example: ./asr/models/CrisperWhisper*

6. Run the model by using the python script `whisper.py`

# Lite-Whisper
### Notes:
- **Partitioning**: Uses **hard segmentation** (no overlap/stride).
- The processor must be downloaded from [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3).

## Instructions
---
1. Create environment with Python 3.10 (you can use the existing `nemo` conda environment):
2. Install required dependencies:
```bash
pip install torch torchaudio transformers librosa numpy
```
3. Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) (for Speed-up)
4. Login to HuggingFace and download model files
```bash
huggingface-cli login
```
- Download the main model:
```bash
huggingface-cli download efficient-speech/lite-whisper-large-v3  --local-dir [SAVE_DIR]
huggingface-cli download efficient-speech/lite-whisper-large-v3-acc --local-dir [SAVE_DIR]
```
- Downlaod the processor model:
```bash
huggingface-cli download openai/whisper-large-v3 --local-dir [SAVE_DIR]
```
*`[SAVE_DIR]` is your target directory. Inside asr/models. Example: ./asr/models/lite_whisper_large_v3*

5. Run the model by using the python script 'lite_whisper.py'.


