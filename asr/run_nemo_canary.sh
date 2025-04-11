#!/bin/bash

# ==============================================================================
# Nemo Canary Experiment Runner
# ==============================================================================
# This script automates running speech recognition experiments with various 
# configurations and calculates Word Error Rate (WER).
# ==============================================================================

set -eo pipefail  # Exit on error and pipe failures

# Example Usage:
# bash ./run_nemo_canary.sh -d /data/manifest.json -m /models/my_model.nemo \
#     -o /output -c "10 20 30" -b "8 16" -k 2 -s 5
# 
# bash ./run_nemo_canary.sh --model-path /path/my_model.nemo \
#     --dataset-manifest /path/manifest.json --output-dir /output \
#     --chunk-lengths "10 20 30" --batch-sizes "8 16" --beam-size 1


# ==============================================================================
# Global Configuration and Defaults
# ==============================================================================
readonly DEFAULT_PRETRAINED="nvidia/canary-1b-flash"
readonly DEFAULT_OUTPUT="./results"
readonly DEFAULT_CHUNKS=(20 30 60 80)
readonly DEFAULT_BATCHES=(4 3 2)
readonly DEFAULT_BEAM=1
readonly DEFAULT_SLEEP=2

# ==============================================================================
# Initialize variables with default values
# ==============================================================================
model_path=""
dataset_manifest=""
pretrained_name="$DEFAULT_PRETRAINED"
output_base_dir="$DEFAULT_OUTPUT"
chunk_lengths=("${DEFAULT_CHUNKS[@]}")
batch_sizes=("${DEFAULT_BATCHES[@]}")
beam_size="$DEFAULT_BEAM"
sleep_time="$DEFAULT_SLEEP"


# ==============================================================================
# Helper Functions
# ==============================================================================
show_help() {
    cat <<EOF
Usage: $0 [options]

Mandatory Parameters:
  -m, --model-path MODEL_PATH        Path to .nemo model file (must end with .nemo)
  -d, --dataset-manifest MANIFEST    Path to .json manifest file

Optional Parameters:
  -p, --pretrained-name NAME         Pretrained model name (default: nvidia/canary-1b-flash)
  -o, --output-dir OUTPUT_DIR        Base output directory (default: ./results)
  -c, --chunk-lengths LENGTHS        Space-separated chunk lengths (default: "20 30 60 80")
  -b, --batch-sizes SIZES            Space-separated batch sizes (default: "4 3 2")
  -k, --beam-size SIZE               Decoding beam size (default: 1)
  -s, --sleep-time SECONDS           Sleep between runs (default: 2)
  -h, --help                         Show this help message
EOF
  exit 0
}

validate_file() {
    local file_path="$1"
    local extension="$2"
    
    [[ -z "${file_path}" ]] && { echo "ERROR: Empty path provided"; exit 1; }
    [[ ! -f "${file_path}" ]] && { echo "ERROR: File not found: ${file_path}"; exit 1; }
    [[ "${file_path}" =~ \.${extension}$ ]] || { echo "ERROR: Invalid .${extension} file: ${file_path}"; exit 1; }
}

validate_positive_integer() {
    local value="$1"
    local name="$2"
    
    [[ ! "${value}" =~ ^[1-9][0-9]*$ ]] && { 
        echo "ERROR: ${name} must be a positive integer, got '${value}'"
        exit 1
    }
}


# ==============================================================================
# Parameter Parsing
# ==============================================================================
parse_parameters() {
    local parsed_args
    parsed_args=$(getopt -o m:d:p:o:c:b:k:s:h \
                --long model-path:,dataset-manifest:,pretrained-name:,output-dir:,chunk-lengths:,batch-sizes:,beam-size:,sleep-time:,help \
                -n "$0" -- "$@") || { show_help; exit 1; }

    eval set -- "${parsed_args}"

    while true; do
        case "$1" in
            -m|--model-path)
                model_path="$2"
                validate_file "${model_path}" "nemo"
                shift 2 ;;
            -d|--dataset-manifest)
                dataset_manifest="$2"
                validate_file "${dataset_manifest}" "json"                
                shift 2 ;;
            -p|--pretrained-name)
                pretrained_name="$2"
                shift 2 ;;
            -o|--output-dir)
                output_base_dir="$2"
                shift 2 ;;
            -c|--chunk-lengths)
                local trimmed_chunks
                trimmed_chunks=$(echo "$2" | xargs)            
                IFS=' ' read -r -a chunk_lengths <<< "$trimmed_chunks"
                [[ ${#chunk_lengths[@]} -eq 0 ]] && chunk_lengths=("${DEFAULT_CHUNKS[@]}")                
                shift 2 ;;
            -b|--batch-sizes)
                local trimmed_batches
                trimmed_batches=$(echo "$2" | xargs)
                IFS=' ' read -r -a batch_sizes <<< "$trimmed_batches"
                [[ ${#batch_sizes[@]} -eq 0 ]] && batch_sizes=("${DEFAULT_BATCHES[@]}")                
                shift 2 ;;
            -k|--beam-size)
                beam_size="$2"
                validate_positive_integer "${beam_size}" "Beam size"                
                shift 2 ;;
            -s|--sleep-time)
                sleep_time="$2"
                validate_positive_integer "${sleep_time}" "Sleep time"                
                shift 2 ;;
            -h|--help) show_help ;;
            --) shift; break ;;
            *) echo "Invalid option: $1"; exit 1 ;;
        esac
    done

    # Validate mandatory parameters
    [[ -z "${model_path}" ]] && { echo "ERROR: Missing --model-path"; exit 1; }
    [[ -z "${dataset_manifest}" ]] && { echo "ERROR: Missing --dataset-manifest"; exit 1; }
    [[ ${#chunk_lengths[@]} -eq 0 ]] && {echo "ERROR: No chunk lengths specified"; exit 1; }
    [[ ${#batch_sizes[@]} -eq 0 ]] && {echo "ERROR: No batch sizes specified"; exit 1; }
}    


# ==============================================================================
# Experiment Setup
# ==============================================================================
create_experiment_dir() {
    local model_dir_name="${pretrained_name//\//_}"
    local experiment_dir="${output_base_dir}/${model_dir_name}"
    
    mkdir -p "${experiment_dir}" || {
        echo "ERROR: Failed to create output directory: ${experiment_dir}"
        exit 1
    }
    echo "${experiment_dir}"
}

initialize_log_file() {
    local log_path="$1"
    cat <<EOF > "${log_path}"
=== Experiment Parameters ===
Start Time:    $(date +'%Y-%m-%d %H:%M:%S')
Model Path:    ${model_path}
Pretrained:    ${pretrained_name}
Dataset:       ${dataset_manifest}
Chunk Lengths: "${chunk_lengths[*]}"
Batch Sizes:   "${batch_sizes[*]}"
Beam Size:     ${beam_size}
Sleep Time:    ${sleep_time}
=============================
EOF
}

# ==============================================================================
# Experiment Execution
# ==============================================================================
run_experiment() {
    local batch_size="$1"
    local chunk_len="$2"
    local experiment_dir="$3"
    local model_dir_name=$(basename "${experiment_dir}")
    local timestamp=$(date +'%Y%m%d_%H%M%S')

    local log_filename="${model_dir_name}_chunk${chunk_len}_beam${beam_size}_batch${batch_size}_${timestamp}.log"
    local log_file="${experiment_dir}/${log_filename}"
    local output_file="${experiment_dir}/${model_dir_name}_chunk${chunk_len}_beam${beam_size}_batch${batch_size}_${timestamp}.json"

    echo "Running configuration:"
    echo "  - Chunk length: ${chunk_len}s"
    echo "  - Batch size: ${batch_size}"
    echo "  - Beam size: ${beam_size}"
    echo "  - Log file: ${log_filename}"
    
    # Initialize log with header
    initialize_log_file "${log_file}"

    {
        echo -e "\n=== Starting Transcription ===\n"
        python nemo_to_text_aed_chunked_infer.py \
            model_path="${model_path}" \
            pretrained_name="${pretrained_name}" \
            dataset_manifest="${dataset_manifest}" \
            output_filename="${output_file}" \
            chunk_len_in_secs="${chunk_len}.0" \
            batch_size="${batch_size}" \
            decoding.beam.beam_size="${beam_size}" || {
                echo "ERROR: Transcription failed for chunk ${chunk_len}, batch ${batch_size}"
                exit 1
            }

        echo -e "\n=== Calculating WER ===\n"
        python ./asr/new_wer_calculator.py \
            -i "${output_file}" \
            -v || {
                echo "ERROR: WER calculation failed for ${output_file}"
                exit 1
            }

        echo -e "\n=== Sleeping for ${sleep_time}s ===\n"
        sleep "${sleep_time}"
    } 2>&1 | tee -a "${log_file}"
}

# ==============================================================================
# Main Program
# ==============================================================================
main() {
    parse_parameters "$@"
    
    local experiment_dir=$(create_experiment_dir)
    local total_configs=$((${#batch_sizes[@]} * ${#chunk_lengths[@]}))
    
    echo -e "\n=== Starting Experiment Series ==="
    echo "Model:          ${pretrained_name}"
    echo "Output Directory: ${experiment_dir}"
    echo "Total Configurations: ${total_configs}"
    echo "----------------------------------------"

    local count=1
    for batch_size in "${batch_sizes[@]}"; do
        validate_positive_integer "${batch_size}" "Batch size"
        for chunk_len in "${chunk_lengths[@]}"; do
            validate_positive_integer "${chunk_len}" "Chunk length"
            echo -e "\n=== Processing Configuration ${count}/${total_configs} ==="
            run_experiment "${batch_size}" "${chunk_len}" "${experiment_dir}"
            ((count++))
        done
    done

    echo -e "\n=== All Experiments Completed ==="
    echo "Final results available in: ${experiment_dir}"
}

main "$@"