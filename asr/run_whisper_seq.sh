#!/bin/bash

# ==============================================================================
# CrisperWhisper Experiment Runner
# ==============================================================================
# This script automates running speech recognition experiments with various 
# configurations and calculates Word Error Rate (WER).
# ==============================================================================
# set -x
# set -eo pipefail  # Exit on error and pipe failures

# Example Usage:
# bash run_crisperwhisper.sh -m /models/CrisperWhisper -i /audio_files \
#     -o /output -c "20 30" -b "2 4" -t "none" -e ".wav"
#
# bash run_crisperwhisper.sh \
#   --model /home/ext_alzahidy_misk_mayo_edu/speech_ker/asr/models/crisperwhisper_model \
#   --input-dir /home/ext_alzahidy_misk_mayo_edu/speech_ker/audio_files/audio_valid \
#   --output-dir /home/ext_alzahidy_misk_mayo_edu/speech_ker/asr/output \
#   --batch-sizes "2 4" \
#   --timestamp "none" \
#   --extensions ".wav" \
#   --sleep-time 10


# ==============================================================================
# Global Configuration and Defaults
# ==============================================================================
readonly DEFAULT_OUTPUT="./results"
readonly DEFAULT_CHUNKS=(20 30)
readonly DEFAULT_BATCHES=(4 2)
readonly DEFAULT_TIMESTAMP="none"
readonly DEFAULT_EXTENSION=".wav"
readonly DEFAULT_SLEEP=2

condition_on_prev_tokens=""
# ==============================================================================
# Initialize variables with default values
# ==============================================================================
model=""
input_dir=""
output_base_dir="$DEFAULT_OUTPUT"
batch_sizes=("${DEFAULT_BATCHES[@]}")
timestamp="$DEFAULT_TIMESTAMP"
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
  -b, --batch-sizes SIZES           Space-separated batch sizes (default: "4 2")
  -t, --timestamp TYPE              Timestamp type ("word", "segment", or "none") (default: none)
  -e, --extensions EXT              File extension to process (default: ".wav")
  -c, --condition-on-prev-tokens    Enable “condition_on_prev_tokens” in generation (default: off)
  -s, --sleep-time SECONDS          Sleep between runs (default: 2)
  -h, --help                        Show this help message
EOF
  exit 0
}

validate_directory() {
    local dir_path="$1"
    local name="$2"
    
    [[ -z "${dir_path}" ]] && { 
        echo "ERROR: Empty ${name} path provided"
        exit 1
    }
    
    # Detailed diagnostics
    echo "=== Directory Validation ==="
    echo "Checking: ${dir_path}"
    ls -ld "${dir_path}" 2>&1 | sed 's/^/DEBUG: /'
    [[ ! -d "${dir_path}" ]] && { 
        echo "ERROR: Directory not found: ${dir_path}"
        echo "Parent directory contents:"
        ls -la "${dir_path%/*}"
        exit 1
    }
    echo "Validation passed"
}

validate_positive_integer() {
    local value="$1"
    local name="$2"
    
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then 
        echo "ERROR: ${name} must be a positive integer, got '${value}'"
        exit 1
    fi
}

validate_timestamp() {
    local ts="$1"
    case "$ts" in
        "word"|"segment"|"none") ;;
        *) echo "ERROR: Invalid timestamp value '$ts'. Must be 'word', 'segment', or 'none'"; exit 1 ;;
    esac
}

# ==============================================================================
# Parameter Parsing
# ==============================================================================
parse_parameters() {
    local parsed_args
    parsed_args=$(getopt -o m:i:o:b:t:e:s:ch \
                --long model:,input-dir:,output-dir:,batch-sizes:,timestamp:,extensions:,sleep-time:,condition-on-prev-tokens,help \
                -n "$0" -- "$@") || { show_help; exit 1; }
    # parsed_args=$(/usr/local/opt/gnu-getopt/bin/getopt \
    # -o m:i:o:c:b:t:e:s:h \
    # --long model:,input-dir:,output-dir:,chunk-lengths:,batch-sizes:,timestamp:,extensions:,sleep-time:,help \
    # -n "$0" -- "$@") || { show_help; exit 1; }
    
    eval set -- "${parsed_args}"

    while true; do
        case "$1" in
            -m|--model)
                model="$2"
                shift 2 ;;
            -i|--input-dir)
                input_dir="$(realpath "$2")"  # Use realpath for absolute paths
                validate_directory "${input_dir}" "Input"
                shift 2 ;;
            -o|--output-dir)
                output_base_dir="$2"
                shift 2 ;;
            -b|--batch-sizes)
                IFS=' ' read -r -a batch_sizes <<< "$(echo "$2" | xargs)"
                [[ ${#batch_sizes[@]} -eq 0 ]] && batch_sizes=("${DEFAULT_BATCHES[@]}")
                shift 2 ;;
            -t|--timestamp)
                timestamp="$2"
                validate_timestamp "${timestamp}"
                shift 2 ;;
            -e|--extensions)
                extensions="$2"
                shift 2 ;;
            -s|--sleep-time)
                sleep_time="$2"
                validate_positive_integer "${sleep_time}" "Sleep time"
                shift 2 ;;
            -c|--condition-on-prev-tokens)
                condition_on_prev_tokens="--condition_on_prev_tokens"
                shift ;;
            -h|--help) show_help ;;
            --) shift; break ;;
            *) echo "Invalid option: $1"; exit 1 ;;
        esac
    done

    # Validate mandatory parameters
    [[ -z "${model}" ]] && { echo "ERROR: Missing --model"; exit 1; }
    [[ -z "${input_dir}" ]] && { echo "ERROR: Missing --input-dir"; exit 1; }
}

# ==============================================================================
# Experiment Setup
# ==============================================================================
create_experiment_dir() {

    model="${model%/}"  # Remove trailing slash (if present)
    local model_dir_name=$(basename "${model}")  # Extract only the directory name    
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
Batch Sizes:   "${batch_sizes[*]}"
Timestamp:     "${timestamp}"
Extensions:    "${extensions}"
Sleep Time:    ${sleep_time}
Condition on Previous Tokens: ${condition_on_prev_tokens:+Yes}
=============================
EOF
}

# ==============================================================================
# Experiment Execution
# ==============================================================================
run_experiment() {
    local batch_size="$1"
    local experiment_dir="$2"
    local model_dir_name=$(basename "${experiment_dir}")
    local output_filename="${model_dir_name}_b${batch_size}"

    local log_file="${experiment_dir}/${output_filename}.log"

    echo "Running configuration:"
    echo "  - Batch size: ${batch_size}"
    echo "  - Timestamp: ${timestamp}"
    echo "  - Condition on previous tokens: ${condition_on_prev_tokens}"
    echo "  - Output filename: ${output_filename}"
    echo "  - Input for new_wer_calculator.py: ${model_dir_name}/${output_filename}.json"
    # === FIX 2: Create directory for log ===
    mkdir -p "${experiment_dir}" || {
        echo "ERROR: Failed to create experiment directory"
        exit 1
    }
    initialize_log_file "${log_file}"

    {
        echo -e "\n=== Starting Transcription ===\n"
        python -u whisper_seq.py \
            --input_dir "${input_dir}" \
            --output_dir "${experiment_dir}" \
            --output_filename "${output_filename}" \
            --model "${model}" \
            --batch_size "${batch_size}" \
            --timestamps "${timestamp}" \
            --extensions "${extensions}" \
            $condition_on_prev_tokens \
            --gold_standard || {
                echo "ERROR: Transcription failed for batch ${batch_size}"
                exit 1
            }


        echo -e "\n=== Calculating WER ===\n"
        python -u new_wer_calculator.py \
            -i "${experiment_dir}/${output_filename}.json" \
            -v || {
                echo "ERROR: WER calculation failed for ${experiment_dir}/${output_filename}.json"
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
    local total_configs=$((${#batch_sizes[@]}))
    
    echo -e "\n=== Starting Experiment Series ==="
    echo "Model:          ${model}"
    echo "Input Directory: ${input_dir}"
    echo "Output Directory: ${experiment_dir}"
    echo "Timestamp Type: ${timestamp}"
    echo "Condition on Prev Tokens: ${condition_on_prev_tokens:+Yes}"
    echo "Total Configurations: ${total_configs}"
    echo "----------------------------------------"

    local count=1
    for batch_size in "${batch_sizes[@]}"; do
        validate_positive_integer "${batch_size}" "Batch size"
        echo -e "\n=== Processing Configuration ${count}/${total_configs} ==="
        run_experiment "${batch_size}" "${experiment_dir}"
        ((count++))
        
    done

    echo -e "\n=== All Experiments Completed ==="
    echo "Final results available in: ${experiment_dir}"
}

main "$@"
