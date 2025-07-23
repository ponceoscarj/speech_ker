# nvidia NeMo Models


## Manifest file
The input of all NeMo ASR models is a manifest file that has a `json` format. This manifest can be generated with the `create_manifest.py` file.

Every line of the `json` file contains the absolute path of the audio file to be transcribed with additional information. Example:

```bash 
{"audio_filepath": "/abs_path/to/audio_file.wav", 
"duration": null, # this is null 
"taskname": "asr", # the task is ASR
"source_lang": "en", # English
"target_lang": "en",  # English
"pnc": "yes", # output with punctuation yes/no
"text": "hello doctor. hello will, i hope you had a good day ...."} # Gold-standard transcription
```

### Manifest file generation
Generate the manifest file with `create_manifest.py` within this folder.

```bash
# Example
python create_manifest.py asr \
    --data_dir /abs_path/to/audio_files_&_txt_files \
    --work_dir /abs_path/to/save/manifest_file \
    --output_file asr_manifest \
    --source_lang en \
    --target_lang en \
    --include_ground_truth \
    --pnc yes
 ```

## Model Characteristics
Brief description of the models from NVIDIA NeMo.



## Manifest generation
All NeMo ASR models need a manifest file which can be created with the `create_manifest.py` script. Instructions for manifest generation for ASR models are within the script. 

Introduce this manifest when running the models under the `dataset_manifest` parameter.



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

