# nvidia NeMo Models

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

