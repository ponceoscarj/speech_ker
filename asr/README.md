Benchmark of ASR models
==============

# NVIDIA NeMo Models

## Model Characteristics
Brief description of the models from NVIDIA NeMo.

### Transformer (AED) Models
| Model Name                  | Encoder          | Decoder      | NeMo Call             | Python Script         |
|-----------------------------|------------------|--------------|-----------------------|-----------------------|
| `nvidia/canary-1b-flash`    | FastConformer    | Transformer  | EncDecMultiTaskModel  | `aed_chunked_infer`   |
| `nvidia/canary-1b`          | FastConformer    | Transformer  | EncDecMultiTaskModel  | `aed_chunked_infer`   |
| `nvidia/canary-180m-flash`  | FastConformer    | Transformer  | EncDecMultiTaskModel  | `aed_chunked_infer`   |

### RNNT Models
| Model Name                              | Encoder          | Decoder      | NeMo Call             | Python Script         |
|-----------------------------------------|------------------|--------------|-----------------------|-----------------------|
| `nvidia/parakeet-tdt-1.1b`             | FastConformer    | RNNT loss    | EncDecRNNTBPEModel    | `buffered_infer_rnnt` |
| `nvidia/parakeet-rnnt-1.1b`            | FastConformer    | RNNT loss    | EncDecRNNTBPEModel    | `buffered_infer_rnnt` |
| `nvidia/parakeet-rnnt-0.6b`            | FastConformer    | RNNT loss    | EncDecRNNTBPEModel    | `buffered_infer_rnnt` |
| `nvidia/stt_en_fastconformer_transducer_large` | FastConformer | RNNT loss    | EncDecRNNTBPEModel    | `buffered_infer_rnnt` |
| `stt_en_conformer_transducer_small`    | Conformer        | RNNT loss    | EncDecRNNTBPEModel    | `buffered_infer_rnnt` |

### CTC Models
| Model Name                              | Encoder          | Decoder      | NeMo Call             | Python Script         |
|-----------------------------------------|------------------|--------------|-----------------------|-----------------------|
| `nvidia/parakeet-ctc-1.1b`             | FastConformer    | CTC loss     | EncDecCTCModelBPE     | `buffered_infer_ctc`  |
| `nvidia/parakeet-ctc-0.6b`             | FastConformer    | CTC loss     | EncDecCTCModelBPE     | `buffered_infer_ctc`  |
| `nvidia/stt_en_conformer_ctc_large`    | Conformer        | CTC loss     | EncDecCTCModelBPE     | `buffered_infer_ctc`  |
| `nvidia/stt_en_fastconformer_ctc_large`| FastConformer    | CTC loss     | EncDecCTCModelBPE     | `buffered_infer_ctc`  |
| `nvidia/stt_en_conformer_ctc_small`    | Conformer        | CTC loss     | EncDecCTCModelBPE     | `buffered_infer_ctc`  |

### Hybrid Models
| Model Name                   | Encoder          | Decoder       | NeMo Call       | Python Script         |
|------------------------------|------------------|---------------|-----------------|-----------------------|
| `nvidia/parakeet-tdt_ctc-110m` | FastConformer   | Hybrid TDT-CTC | ASRModel?       | `buffered_infer_rnnt` |


## Instructions
Create conda environment with python 3.10.12 - more stable
```console
conda create --name nemo python==3.10.12
conda activate nemo
```

Install latest torch - check your CUDA version [here](https://pytorch.org/get-started/locally/)

```console
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install NeMO from main branch + dependencies
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packagin
python -m pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]'
```
## Manifest generation
All ASR models need to run with a manifest file. This is latter introduced in the `dataset_manifest` parameter.

Use the `create_manifest.py` code and the `type` parameter should be `asr`.

## Download models
You can

## General Parameters
- `model_path`: Local path to .nemo file 
- `pretrained_name`: Model name
- `audio_dir`: Directory containing WAV files
- `dataset_manifest`: Path to the manifest file
- `output_filename`: Output results path

### CTC Models - buffered_infer_ctc

```bash
python speech_to_text_buffered_infer_ctc.py \
    model_path="stt_en_conformer_ctc_small.nemo" \
    pretrained_name="stt_en_conformer_ctc_small" \
    audio_dir="/path/to/audio_files" \
    dataset_manifest="/path/to/manifest.json" \
    output_filename="/output/path/results.json" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=1 \
    clean_groundtruth_text=True \
    langid='en'
```


### RNNT Model Example - buffered_infer_rnnt

#### Middle Token Merge
```bash
python speech_to_text_buffered_infer_rnnt.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
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
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=32 \
    merge_algo="lcs" \
    lcs_alignment_dir=<OPTIONAL: Some path to store the LCS alignments> 
```


### AED Example - aed_chunked_infer

```console
python speech_to_text_aed_chunked_infer.py \
    model_path=null \
    pretrained_name="nvidia/canary-1b-flash" \
    audio_dir="<(optional) path to folder of audio files>" \
    dataset_manifest="<(optional) path to manifest>" \
    output_filename="<(optional) specify output filename>" \
    chunk_len_in_secs=40.0 \
    batch_size=16 \
    decoding.beam.beam_size=1
```
