Benchmark of ASR models
==============

The list of models to be evaluated are available on [Huggingface open asr leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)


# NVIDIA NeMo Models

## Model Characteristics
Brief description of the models from NVIDIA NeMo.

### Transformer (AED) Models
| Model Name                  | Encoder          | Decoder      | NeMo Call             | Python Script         |
|-----------------------------|------------------|--------------|-----------------------|-----------------------|
| `nvidia/canary-1b-flash`    | FastConformer    | Transformer  | EncDecMultiTaskModel  | `nemo_aed_chunked_infer`   |
| `nvidia/canary-1b`          | FastConformer    | Transformer  | EncDecMultiTaskModel  | `nemo_aed_chunked_infer`   |
| `nvidia/canary-180m-flash`  | FastConformer    | Transformer  | EncDecMultiTaskModel  | `nemo_aed_chunked_infer`   |

### RNNT Models
| Model Name                              | Encoder          | Decoder      | NeMo Call             | Python Script         |
|-----------------------------------------|------------------|--------------|-----------------------|-----------------------|
| `nvidia/parakeet-tdt-1.1b`             | FastConformer    | RNNT loss    | EncDecRNNTBPEModel    | `nemo_buffered_infer_rnnt` |
| `nvidia/parakeet-rnnt-1.1b`            | FastConformer    | RNNT loss    | EncDecRNNTBPEModel    | `nemo_buffered_infer_rnnt` |
| `nvidia/parakeet-rnnt-0.6b`            | FastConformer    | RNNT loss    | EncDecRNNTBPEModel    | `nemo_buffered_infer_rnnt` |
| `nvidia/stt_en_fastconformer_transducer_large` | FastConformer | RNNT loss    | EncDecRNNTBPEModel    | `nemo_buffered_infer_rnnt` |
| `stt_en_conformer_transducer_small`    | Conformer        | RNNT loss    | EncDecRNNTBPEModel    | `nemo_buffered_infer_rnnt` |

### CTC Models
| Model Name                              | Encoder          | Decoder      | NeMo Call             | Python Script         |
|-----------------------------------------|------------------|--------------|-----------------------|-----------------------|
| `nvidia/parakeet-ctc-1.1b`             | FastConformer    | CTC loss     | EncDecCTCModelBPE     | `nemo_buffered_infer_ctc`  |
| `nvidia/parakeet-ctc-0.6b`             | FastConformer    | CTC loss     | EncDecCTCModelBPE     | `nemo_buffered_infer_ctc`  |
| `nvidia/stt_en_conformer_ctc_large`    | Conformer        | CTC loss     | EncDecCTCModelBPE     | `nemo_buffered_infer_ctc`  |
| `nvidia/stt_en_fastconformer_ctc_large`| FastConformer    | CTC loss     | EncDecCTCModelBPE     | `nemo_buffered_infer_ctc`  |
| `nvidia/stt_en_conformer_ctc_small`    | Conformer        | CTC loss     | EncDecCTCModelBPE     | `nemo_buffered_infer_ctc`  |

### Hybrid Models
| Model Name                   | Encoder          | Decoder       | NeMo Call       | Python Script         |
|------------------------------|------------------|---------------|-----------------|-----------------------|
| `nvidia/parakeet-tdt_ctc-110m` | FastConformer   | Hybrid TDT-CTC | ASRModel?       | `nemo_buffered_infer_rnnt` |


## Requirements
1. Create conda environment with python 3.10.12 - more stable
```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

2. Install latest torch - check your CUDA version [here](https://pytorch.org/get-started/locally/)
```bash
# The CUDA version of the sandbox-AI is 12.4
# you can check cuda version with nvidia-smi
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

3. Install NeMo important dependencies
```bash
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
```

4. Install NeMo
```bash
python -m pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]'
```
OR
```bash
pip install "nemo_toolkit[all]"===999 #version 999 does not exist but it will trigger nemo_toolkit to show you all the version, choose the latest.
pip install "nemo_toolkit[all]"===2.3.0rc2 #this is the latest version as of April 11th, 2025
```

## Manifest generation
All NeMo ASR models need a manifest file which can be created with the `create_manifest.py` script. Instructions for manifest generation for ASR models are within the script. 

Introduce this manifest when running the models under the `dataset_manifest` parameter.

## Download NeMo models
1. Install the huggingface_hub package:
```bash
pip install -U "huggingface_hub[cli]"
```

2. Download models form huggingface: 
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


# CrisperWhisper

## Instructions
1. Create environment with Python 3.10 (more stable) - use your nemo conda environment if you have it
 
2. Clone repository into your `asr/models` folder:
```bash
git clone https://github.com/nyrahealth/CrisperWhisper.git
cd CrisperWhisper
```

3. Install the dependencies of CrisperWhisper and `crisperwhisper.py`
 ```bash
 pip install -r ./CrisperWhisper/requirements.txt
 pip install jiwer # for WER calculation
 ```

4. Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) (for Speed-up):
   
**Requirements**:
```yaml
- CUDA toolkit or ROCm toolkit:     *(Install manually according to your GPU)*
- PyTorch 2.2 or above: 		    Already met with nemo conda environment
- `packaging`:                      `pip install packaging`
- `ninja`:                          `pip install ninja`
```
**Install FlashAttention:**
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
 ```

5. Install the CrisperWhiser model

- Accept the license of the model [CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper)
- Login into huggingface and introduce your token. 
```bash
huggingface-cli login
```
- Download model files to a specific folder:
```bash
huggingface-cli download nyrahealth/CrisperWhisper --local-dir [SAVE_DIR]
```
*`[SAVE_DIR]` is your target directory. Inside asr/models. Example: ./asr/models/CrisperWhisper*

6. Run the model by using the python script `crisperwhisper.py`

# Lite-Whisper

### Notes:
- **Partitioning**: Uses **hard segmentation** (no overlap/stride).
- **FlashAttention is NOT required** for this model.
- The processor must be downloaded from [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3).


| Model Name                                  | Processor Model               | Description                         | Python Script Name                          |
|---------------------------------------------|-------------------------------|-------------------------------------|--------------------------------------|
| `efficient-speech/lite-whisper-large-v3`    | `openai/whisper-large-v3`     | Base lightweight Whisper variant    | `lite_whisper_large_v3.py`           |
| `efficient-speech/lite-whisper-large-v3-acc`| `openai/whisper-large-v3`     | Accuracy-tuned variant              | `lite_whisper_large_v3_acc.py`       |


## Instructions

---
1. Create environment with Python 3.10 (you can use the existing `nemo` conda environment):
2. Install required dependencies:
```bash
pip install torch torchaudio transformers librosa numpy
```
3. Login to HuggingFace and download model files
- Login into huggingface and introduce your token. 
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

4. Run the model by using the python appropriate python script.


