#!/bin/bash

# ==============================================================================
# Nemo Buffered RNNT Experiment Runner
# ==============================================================================
# This script automates running buffered RNNT inference experiments with various 
# configurations and calculates Word Error Rate (WER).
# ==============================================================================
set -x
set -eo pipefail  # Exit on error and pipe failures

# Example Usage:
# bash run_nemo_buffered.sh --model-path /path/to/model.nemo \
#     --dataset-manifest /path/to/manifest.json 
#     --output-dir /output \
#     --total-buffers "4.0 6.0" --chunk-lengths "1.6 2.0" --batch-sizes "32 16" \
#     --model-stride 4 --merge-algo "lcs"

# ==============================================================================
# Global Configuration and Defaults
# ==============================================================================
readonly DEFAULT_PRETRAINED="nvidia/canary-1b-flash"
readonly DEFAULT_OUTPUT="./results"
readonly DEFAULT_TOTAL_BUFFERS=(4.0 6.0)
readonly DEFAULT_CHUNK_LENS=(1.6 2.0)
readonly DEFAULT_BATCH_SIZES=(32 16)
readonly DEFAULT_MODEL_STRIDE=4
readonly DEFAULT_MERGE_ALGO="middle"
readonly DEFAULT_SLEEP=2

# ==============================================================================
# Initialize variables with default values
# ==============================================================================
model_path=""
dataset_manifest=""
pretrained_name="$DEFAULT_PRETRAINED"
output_base_dir="$DEFAULT_OUTPUT"
total_buffers=("${DEFAULT_TOTAL_BUFFERS[@]}")
chunk_lens=("${DEFAULT_CHUNK_LENS[@]}")
batch_sizes=("${DEFAULT_BATCH_SIZES[@]}")
model_stride="$DEFAULT_MODEL_STRIDE"
merge_algo="$DEFAULT_MERGE_ALGO"
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
  -p, --pretrained-name NAME         Pretrained model name (default: $DEFAULT_PRETRAINED)
  -o, --output-dir OUTPUT_DIR        Base output directory (default: $DEFAULT_OUTPUT)
  -t, --total-buffers BUFFERS        Space-separated total buffer sizes in seconds (default: "${DEFAULT_TOTAL_BUFFERS[@]}"). Length of buffer (chunk + left and right padding) in seconds.
  -c, --chunk-lengths LENGTHS        Space-separated chunk lengths in seconds (default: "${DEFAULT_CHUNK_LENS[@]}")
  -b, --batch-sizes SIZES            Space-separated batch sizes (default: "${DEFAULT_BATCH_SIZES[@]}")
  -r, --model-stride STRIDE          Model stride factor (default: $DEFAULT_MODEL_STRIDE)
  -a, --merge-algo ALGO              Merge algorithm: 'middle' or 'lcs' (default: $DEFAULT_MERGE_ALGO)
  -s, --sleep-time SECONDS           Sleep between runs (default: $DEFAULT_SLEEP)
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

validate_positive_number() {
    local value="$1"
    local name="$2"
    
    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! (( $(echo "$value >= 0" | bc -l) )); then 
        echo "ERROR: ${name} must be a positive number, got '${value}'"
        exit 1
    fi
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
    parsed_args=$(getopt -o m:d:p:o:t:c:b:r:a:s:h \
                --long model-path:,dataset-manifest:,pretrained-name:,output-dir:,total-buffers:,chunk-lengths:,batch-sizes:,model-stride:,merge-algo:,sleep-time:,help \
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
            -t|--total-buffers)
                local trimmed_buffers
                trimmed_buffers=$(echo "$2" | xargs)
                IFS=' ' read -r -a total_buffers <<< "$trimmed_buffers"
                for buf in "${total_buffers[@]}"; do
                    validate_positive_number "${buf}" "Total buffer"
                done
                shift 2 ;;
            -c|--chunk-lengths)
                local trimmed_chunks
                trimmed_chunks=$(echo "$2" | xargs)
                IFS=' ' read -r -a chunk_lens <<< "$trimmed_chunks"
                for chunk in "${chunk_lens[@]}"; do
                    validate_positive_number "${chunk}" "Chunk length"
                done
                shift 2 ;;
            -b|--batch-sizes)
                local trimmed_batches
                trimmed_batches=$(echo "$2" | xargs)
                IFS=' ' read -r -a batch_sizes <<< "$trimmed_batches"
                for bs in "${batch_sizes[@]}"; do
                    validate_positive_integer "${bs}" "Batch size"
                done
                shift 2 ;;
            -r|--model-stride)
                model_stride="$2"
                validate_positive_integer "${model_stride}" "Model stride"
                shift 2 ;;
            -a|--merge-algo)
                merge_algo="$2"
                if [[ "${merge_algo}" != "middle" && "${merge_algo}" != "lcs" ]]; then
                    echo "ERROR: Merge algorithm must be 'middle' or 'lcs'"
                    exit 1
                fi
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

    # Validate array parameters
    if [[ ${#total_buffers[@]} -eq 0 ]]; then
        echo "ERROR: No total buffers specified"
        exit 1
    fi

    if [[ ${#chunk_lens[@]} -eq 0 ]]; then
        echo "ERROR: No chunk lengths specified"
        exit 1
    fi
    if [[ ${#batch_sizes[@]} -eq 0 ]]; then
        echo "ERROR: No batch sizes specified"
        exit 1
    fi
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
    local total_buffer="$2"
    local chunk_len="$3"
    local batch_size="$4"
    cat <<EOF > "${log_path}"
=== Experiment Parameters ===
Start Time:    $(date +'%Y-%m-%d %H:%M:%S')
Model Path:    ${model_path}
Pretrained:    ${pretrained_name}
Dataset:       ${dataset_manifest}
Total Buffer:  ${total_buffer}s
Chunk Length:  ${chunk_len}s
Batch Size:    ${batch_size}
Model Stride:  ${model_stride}
Merge Algo:    ${merge_algo}
Sleep Time:    ${sleep_time}s
=============================
EOF
}

# ==============================================================================
# Experiment Execution
# ==============================================================================
run_experiment() {
    local total_buffer="$1"
    local chunk_len="$2"
    local batch_size="$3"
    local experiment_dir="$4"
    local model_dir_name=$(basename "${experiment_dir}")
    local timestamp=$(date +'%Y%m%d_%H%M%S')

    # Sanitize float values for filenames
    local total_buffer_sanitized="${total_buffer//./_}"
    local chunk_len_sanitized="${chunk_len//./_}"
    
    local log_filename="${model_dir_name}_buffer${total_buffer_sanitized}_chunk${chunk_len_sanitized}_batch${batch_size}_${merge_algo}_${timestamp}.log"
    local log_file="${experiment_dir}/${log_filename}"
    local output_file="${experiment_dir}/${model_dir_name}_buffer${total_buffer_sanitized}_chunk${chunk_len_sanitized}_batch${batch_size}_${merge_algo}_${timestamp}.json"

    echo "Running configuration:"
    echo "  - Total Buffer: ${total_buffer}s"
    echo "  - Chunk Length: ${chunk_len}s"
    echo "  - Batch Size: ${batch_size}"
    echo "  - Merge Algo: ${merge_algo}"
    echo "  - Log file: ${log_filename}"
    
    # Initialize log with header
    initialize_log_file "${log_file}" "${total_buffer}" "${chunk_len}" "${batch_size}"

    {
        echo -e "\n=== Starting Transcription ===\n"
        python3 nemo_buffered_infer_rnnt.py \
            model_path="${model_path}" \
            pretrained_name="${pretrained_name}" \
            dataset_manifest="${dataset_manifest}" \
            output_filename="${output_file}" \
            total_buffer_in_secs="${total_buffer}" \
            chunk_len_in_secs="${chunk_len}" \
            model_stride="${model_stride}" \
            batch_size="${batch_size}" \
            merge_algo="${merge_algo}" \
            clean_groundtruth_text=true \
            langid='en' || {
                echo "ERROR: Transcription failed for buffer ${total_buffer}, chunk ${chunk_len}, batch ${batch_size}"
                exit 1
            }

        echo -e "\n=== Calculating WER ===\n"
        python3 new_wer_calculator.py \
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
    echo "=== Starting main() ==="    
    parse_parameters "$@"
    echo "=== Parameters parsed ==="
        
    local experiment_dir=$(create_experiment_dir)
    local total_configs=$((${#total_buffers[@]} * ${#chunk_lens[@]} * ${#batch_sizes[@]}))
    
    echo -e "\n=== Starting Experiment Series ==="
    echo "Model:          ${pretrained_name}"
    echo "Output Directory: ${experiment_dir}"
    echo "Total Configurations: ${total_configs}"
    echo "----------------------------------------"

    local count=1
    for total_buffer in "${total_buffers[@]}"; do
        validate_positive_number "${total_buffer}" "Total buffer"
        for chunk_len in "${chunk_lens[@]}"; do
            validate_positive_number "${chunk_len}" "Chunk length"
            for batch_size in "${batch_sizes[@]}"; do
                validate_positive_integer "${batch_size}" "Batch size"
                echo -e "\n=== Processing Configuration ${count}/${total_configs} ==="
                run_experiment "${total_buffer}" "${chunk_len}" "${batch_size}" "${experiment_dir}"
                ((count++))
            done
        done
    done

    echo -e "\n=== All Experiments Completed ==="
    echo "Final results available in: ${experiment_dir}"
}

main "$@"