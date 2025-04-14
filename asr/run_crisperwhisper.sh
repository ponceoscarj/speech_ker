#!/bin/bash

# ==============================================================================
# CrisperWhisper Experiment Runner
# ==============================================================================
# This script automates running speech recognition experiments with various 
# configurations and calculates Word Error Rate (WER).
# ==============================================================================
set -x
set -eo pipefail  # Exit on error and pipe failures

# Example Usage:
# bash run_crisperwhisper.sh -m /models/CrisperWhisper -i /audio_files \
#     -o /output -c "20 30" -b "2 4" -t "word segment none" -e ".wav"

# ==============================================================================
# Global Configuration and Defaults
# ==============================================================================
readonly DEFAULT_OUTPUT="./results"
readonly DEFAULT_CHUNKS=(20 30)
readonly DEFAULT_BATCHES=(4 2)
readonly DEFAULT_TIMESTAMPS=("word" "segment" "none")
readonly DEFAULT_EXTENSION=".wav"
readonly DEFAULT_SLEEP=2

# ==============================================================================
# Initialize variables with default values
# ==============================================================================
model=""
input_dir=""
output_base_dir="$DEFAULT_OUTPUT"
chunk_lengths=("${DEFAULT_CHUNKS[@]}")
batch_sizes=("${DEFAULT_BATCHES[@]}")
timestamps=("${DEFAULT_TIMESTAMPS[@]}")
extensions="$DEFAULT_EXTENSION"
sleep_time="$DEFAULT_SLEEP"

# ==============================================================================
# Helper Functions
# ==============================================================================
show_help() {
    cat <<EOF
Usage: $0 [options]

Mandatory Parameters:
  -m, --model MODEL                 Path to model directory or Hugging Face model name
  -i, --input-dir INPUT_DIR         Directory containing audio files

Optional Parameters:
  -o, --output-dir OUTPUT_DIR       Base output directory (default: ./results)
  -c, --chunk-lengths LENGTHS       Space-separated chunk lengths (default: "20 30")
  -b, --batch-sizes SIZES           Space-separated batch sizes (default: "4 2")
  -t, --timestamps TYPES            Space-separated timestamp types (default: "word segment none")
  -e, --extensions EXT              File extension to process (default: ".wav")
  -s, --sleep-time SECONDS          Sleep between runs (default: 2)
  -h, --help                        Show this help message
EOF
  exit 0
}

validate_directory() {
    local dir_path="$1"
    local name="$2"
    
    [[ -z "${dir_path}" ]] && { echo "ERROR: Empty ${name} path provided"; exit 1; }
    [[ ! -d "${dir_path}" ]] && { echo "ERROR: Directory not found: ${dir_path}"; exit 1; }
}

validate_positive_integer() {
    local value="$1"
    local name="$2"
    
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then 
        echo "ERROR: ${name} must be a positive integer, got '${value}'"
        exit 1
    fi
}

# ==============================================================================
# Parameter Parsing
# ==============================================================================
parse_parameters() {
    local parsed_args
    parsed_args=$(getopt -o m:i:o:c:b:t:e:s:h \
                --long model:,input-dir:,output-dir:,chunk-lengths:,batch-sizes:,timestamps:,extensions:,sleep-time:,help \
                -n "$0" -- "$@") || { show_help; exit 1; }

    eval set -- "${parsed_args}"

    while true; do
        case "$1" in
            -m|--model)
                model="$2"
                shift 2 ;;
            -i|--input-dir)
                input_dir="$2"
                validate_directory "${input_dir}" "Input"
                shift 2 ;;
            -o|--output-dir)
                output_base_dir="$2"
                shift 2 ;;
            -c|--chunk-lengths)
                IFS=' ' read -r -a chunk_lengths <<< "$(echo "$2" | xargs)"
                [[ ${#chunk_lengths[@]} -eq 0 ]] && chunk_lengths=("${DEFAULT_CHUNKS[@]}")
                shift 2 ;;
            -b|--batch-sizes)
                IFS=' ' read -r -a batch_sizes <<< "$(echo "$2" | xargs)"
                [[ ${#batch_sizes[@]} -eq 0 ]] && batch_sizes=("${DEFAULT_BATCHES[@]}")
                shift 2 ;;
            -t|--timestamps)
                IFS=' ' read -r -a timestamps <<< "$(echo "$2" | xargs)"
                [[ ${#timestamps[@]} -eq 0 ]] && timestamps=("${DEFAULT_TIMESTAMPS[@]}")
                shift 2 ;;
            -e|--extensions)
                extensions="$2"
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
    [[ -z "${model}" ]] && { echo "ERROR: Missing --model"; exit 1; }
    [[ -z "${input_dir}" ]] && { echo "ERROR: Missing --input-dir"; exit 1; }

    # Validate timestamp values
    for ts in "${timestamps[@]}"; do
        case "$ts" in
            "word"|"segment"|"none") ;;
            *) echo "ERROR: Invalid timestamp value '$ts'"; exit 1 ;;
        esac
    done
}

# ==============================================================================
# Experiment Setup
# ==============================================================================
create_experiment_dir() {
    local model_dir_name="${model//\//_}"
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
Model:         ${model}
Input Dir:     ${input_dir}
Chunk Lengths: "${chunk_lengths[*]}"
Batch Sizes:   "${batch_sizes[*]}"
Timestamps:    "${timestamps[*]}"
Extensions:    "${extensions}"
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
    local timestamp="$3"
    local experiment_dir="$4"
    
    local model_dir_name=$(basename "${experiment_dir}")
    local timestamp_str=$(date +'%Y%m%d_%H%M%S')
    local output_subdir="${experiment_dir}/chunk${chunk_len}_batch${batch_size}_ts_${timestamp}_${timestamp_str}"
    local log_file="${experiment_dir}/${model_dir_name}_chunk${chunk_len}_batch${batch_size}_ts_${timestamp}.log"

    echo "Running configuration:"
    echo "  - Chunk length: ${chunk_len}s"
    echo "  - Batch size: ${batch_size}"
    echo "  - Timestamps: ${timestamp}"
    
    mkdir -p "${output_subdir}"
    initialize_log_file "${log_file}"

    {
        echo -e "\n=== Starting Transcription ===\n"
        python3 crisperwhisper.py \
            --input_dir "${input_dir}" \
            --output_dir "${output_subdir}" \
            --model "${model}" \
            --chunk_length "${chunk_len}" \
            --batch_size "${batch_size}" \
            --timestamps "${timestamp}" \
            --extensions "${extensions}" || {
                echo "ERROR: Transcription failed for chunk ${chunk_len}, batch ${batch_size}, ts ${timestamp}"
                exit 1
            }

        echo -e "\n=== Calculating WER ===\n"
        python3 new_wer_calculator.py \
            -i "${output_subdir}/manifest.json" \
            -v || {
                echo "ERROR: WER calculation failed for ${output_subdir}/manifest.json"
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
    local total_configs=$((${#batch_sizes[@]} * ${#chunk_lengths[@]} * ${#timestamps[@]}))
    
    echo -e "\n=== Starting Experiment Series ==="
    echo "Model:          ${model}"
    echo "Input Directory: ${input_dir}"
    echo "Output Directory: ${experiment_dir}"
    echo "Total Configurations: ${total_configs}"
    echo "----------------------------------------"

    local count=1
    for batch_size in "${batch_sizes[@]}"; do
        validate_positive_integer "${batch_size}" "Batch size"
        for chunk_len in "${chunk_lengths[@]}"; do
            validate_positive_integer "${chunk_len}" "Chunk length"
            for timestamp in "${timestamps[@]}"; do
                echo -e "\n=== Processing Configuration ${count}/${total_configs} ==="
                run_experiment "${batch_size}" "${chunk_len}" "${timestamp}" "${experiment_dir}"
                ((count++))
            done
        done
    done

    echo -e "\n=== All Experiments Completed ==="
    echo "Final results available in: ${experiment_dir}"
}

main "$@"