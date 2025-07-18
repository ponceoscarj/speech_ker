# Benchmark of ASR models

The list of models to be evaluated are available on [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

The architecture characteristics of the models are stated at the end of this `README` file. 

## Requirements
### 1. Create conda environment 
```bash
conda create --name speech_ker python==3.10.12 # more stable
conda activate speech_ker
```

### 2. Install torch 
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
pip install "nemo_toolkit[all]"===2.3.0rc2 #this is the latest version as of April 11th, 2025 - pip install "nemo_toolkit[all]"===2.4.0rc2
```

### 5. Install Huggingface Hub
This is needed to download all ASR models. 
```bash
pip install -U "huggingface_hub[cli]"
```

### 6. Install other dependencies
```
pip install jiwer
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



## Manifest generation
All NeMo ASR models need a manifest file which can be created with the `create_manifest.py` script. Instructions for manifest generation for ASR models are within the script. 

Introduce this manifest when running the models under the `dataset_manifest` parameter.

## Download NeMo models
1. Download models form huggingface: 
Download the `.nemo` file to run the models. 
```bash
huggingface-cli download [REPO_ID] --include [FILE_NAME] --local-dir [SAVE_DIR]
```
- `[REPO_ID]` is the model repository from huggingface. Example `nvidia/parakeet-tdt-1.1b`
- `[FILE_NAME]` path to the file name that ends with `.nemo`. Example `parakeet-tdt-1.1b.nemo`
- `[SAVE_DIR]` is your target directory. Inside asr/models.


## Running the models
1. General parameters for running each model
- `model_path`: Local path to .nemo file 
- `pretrained_name`: Model name
- `audio_dir`: Directory containing WAV files
- `dataset_manifest`: Path to the manifest file
- `output_filename`: Output results path

### Example for CTC Models - nemo_buffered_infer_ctc

```bash
python speech_to_text_buffered_infer_ctc.py \
    model_path="stt_en_conformer_ctc_small.nemo" \
    pretrained_name="stt_en_conformer_ctc_small" \
    dataset_manifest="/path/to/manifest.json" \
    output_filename="/output/path/results.json" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=1 \
    clean_groundtruth_text=True \
    langid='en'
```


### Example for RNNT Models - nemo_buffered_infer_rnnt

#### Middle Token Merge
```bash
python speech_to_text_buffered_infer_rnnt.py \
    model_path=null \
    pretrained_name=null \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=32 \
    clean_groundtruth_text=True \
    langid='en'
```

#### Longer Common Subsequence (LCS) Merge algorithm

```bash
python speech_to_text_buffered_infer_rnnt.py \
    model_path=null \
    pretrained_name=null \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=32 \
    merge_algo="lcs" \
    lcs_alignment_dir=<OPTIONAL: Some path to store the LCS alignments> 
```


### Example for Canary Models - nemo_aed_chunked_infer

```bash
python speech_to_text_aed_chunked_infer.py \
    model_path=null \
    pretrained_name="nvidia/canary-1b-flash" \
    dataset_manifest="<(optional) path to manifest>" \
    output_filename="<(optional) specify output filename>" \
    chunk_len_in_secs=40.0 \
    batch_size=16 \
    decoding.beam.beam_size=1
```


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


