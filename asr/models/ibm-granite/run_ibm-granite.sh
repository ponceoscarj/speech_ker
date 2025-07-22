#!/bin/bash

# ==============================================================================
# Phi-4-multimodal-instruct Experiment Runner
# ==============================================================================
# set -x
# set -eo pipefail  # Exit on error and pipe failures

# Example Usage:
#
# bash run_ibm-granite.sh \
#   --model /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models_outputs \
#   --input-dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/audio_files/toy \
#   --output-dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/output \
#   --system-prompt /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/ibm-granite/system-prompt_ibm-granite.txt \
#   --user-prompt /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/ibm-granite/user-prompt_ibm-granite.txt \
#   --chunk-lengths "20 30" \
#   --batch-sizes "1" \
#   --extensions ".wav" \
#   --sleep-time 10


# ==============================================================================
# Global Configuration and Defaults
# ==============================================================================
readonly DEFAULT_OUTPUT="./results"
readonly DEFAULT_CHUNKS=(20 30)
readonly DEFAULT_BATCHES=(4 2)
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
extensions="$DEFAULT_EXTENSION"
sleep_time="$DEFAULT_SLEEP"
system_prompt=""
user_prompt=""


# ==============================================================================
# Helper Functions
# ==============================================================================
show_help() {
    cat <<EOF
Usage: $0 [options]

Mandatory Parameters:
  -m, --model MODEL               Local path to the ASR model directory
  -i, --input-dir INPUT_DIR       Directory containing audio files
  -s, --system-prompt FILE        System prompt file
  -u, --user-prompt FILE          User prompt file

Optional Parameters:
  -o, --output-dir OUTPUT_DIR     Base output directory (default: ./results)
  -c, --chunk-lengths LENGTHS     Space-separated chunk lengths in seconds (default: 30)
  -b, --batch-sizes SIZES         Space-separated batch sizes (default: 1)
  -e, --extensions EXT            Audio extension(s) to process (default: ".wav")
  -t, --sleep-time SECONDS        Sleep between runs (default: 0)
  -h, --help                      Show this help message
EOF
    exit 0
}


validate_directory() {
    local dir_path="$1"
    local name="$2"
    [[ -z "$dir_path" ]] && { echo "ERROR: Empty $name path"; exit 1; }
    [[ ! -d "$dir_path" ]] && { echo "ERROR: Directory not found: $dir_path"; exit 1; }
}

validate_file() {
    local file_path="$1"
    [[ ! -f "$file_path" ]] && { echo "ERROR: File not found: $file_path"; exit 1; }
}

validate_positive_integer() {
    local value="$1"
    [[ ! "$value" =~ ^[1-9][0-9]*$ ]] && { echo "ERROR: Must be positive integer: '$value'"; exit 1; }
}


# ==============================================================================
# Parameter Parsing
# ==============================================================================
parse_parameters() {
    local parsed_args
    parsed_args=$(getopt -o m:i:o:c:b:e:s:u:t:h \
                  --long model:,input-dir:,output-dir:,chunk-lengths:,batch-sizes:,extensions:,system-prompt:,user-prompt:,sleep-time:,help \
                  -n "$0" -- "$@") || { show_help; exit 1; }
    eval set -- "$parsed_args"

    while true; do
        case "$1" in
            -m|--model)
                model="$2"; shift 2 ;;
            -i|--input-dir)
                input_dir="$(realpath "$2")"; validate_directory "$input_dir" "Input"; shift 2 ;;
            -o|--output-dir)
                output_base_dir="$2"; shift 2 ;;
            -c|--chunk-lengths)
                IFS=' ' read -r -a chunk_lengths <<< "$(echo "$2" | xargs)"; shift 2 ;;
            -b|--batch-sizes)
                IFS=' ' read -r -a batch_sizes <<< "$(echo "$2" | xargs)"; shift 2 ;;
            -e|--extensions)
                extensions="$2"; shift 2 ;;
            -s|--system-prompt)
                system_prompt="$2"; validate_file "$system_prompt"; shift 2 ;;
            -u|--user-prompt)
                user_prompt="$2"; validate_file "$user_prompt"; shift 2 ;;
            -t|--sleep-time)
                sleep_time="$2"; validate_positive_integer "$sleep_time"; shift 2 ;;
            -h|--help) show_help ;;
            --) shift; break ;;
            *) echo "Invalid option: $1"; exit 1 ;;
        esac
    done

    [[ -z "$model" ]] && { echo "ERROR: --model is required"; exit 1; }
    [[ -z "$input_dir" ]] && { echo "ERROR: --input-dir is required"; exit 1; }
    [[ -z "$system_prompt" ]] && { echo "ERROR: --system-prompt is required"; exit 1; }
    [[ -z "$user_prompt" ]] && { echo "ERROR: --user-prompt is required"; exit 1; }
}

# ==============================================================================
# Experiment Setup
# ==============================================================================
create_experiment_dir() {
    model="${model%/}"
    local model_dir_name=$(basename "$model")
    local experiment_dir="${output_base_dir}/${model_dir_name}"
    mkdir -p "$experiment_dir" || { echo "ERROR: Cannot create $experiment_dir"; exit 1; }
    echo "$experiment_dir"
}

initialize_log_file() {
    local log_path="$1"
    {
        echo "=== Experiment Parameters ==="
        echo "Start Time:    $(date +'%Y-%m-%d %H:%M:%S')"
        echo "Model:         $model"
        echo "Input Dir:     $input_dir"
        echo "Chunk Lengths: ${chunk_lengths[*]}"
        echo "Batch Sizes:   ${batch_sizes[*]}"
        echo "Extensions:    $extensions"
        echo "System Prompt: $system_prompt"
        echo "User Prompt:   $user_prompt"
        echo "Sleep Time:    $sleep_time"
        echo "============================="
    } > "$log_path"
}

# ==============================================================================
# Experiment Execution
# ==============================================================================
run_experiment() {
    local batch_size="$1"
    local chunk_len="$2"
    local experiment_dir="$3"

    local model_dir_name=$(basename "$experiment_dir")
    local output_filename="${model_dir_name}_b${batch_size}_c${chunk_len}"
    local log_file="${experiment_dir}/${output_filename}.log"

    echo "Running: chunk=${chunk_len}s batch=${batch_size}  log=${log_file}"
    initialize_log_file "$log_file"

    {
        python -u ibm-granite.py \
            --input_dir "$input_dir" \
            --output_dir "$experiment_dir" \
            --model "$model" \
            --output_filename "$output_filename" \
            --chunk_lengths "$chunk_len" \
            --batch_sizes "$batch_size" \
            --extensions "$extensions" \
            --system_prompt "$system_prompt" \
            --user_prompt "$user_prompt" \
            --sleep-time "$sleep_time" \
            --gold_standard || {
                echo "ERROR: Transcription failed"
                exit 1
            }

        echo -e "\n=== Calculating WER ==="
        python -u ../../new_wer_calculator.py \
            --ind_results "${experiment_dir}/${output_filename}.json" \
            --overall_results "${experiment_dir}/${output_filename}_meta.json" -v || {
                echo "ERROR: WER calculation failed"
                exit 1
            }

        echo -e "\n=== Sleeping for ${sleep_time}s ==="
        sleep "$sleep_time"
    } 2>&1 | tee -a "$log_file"
}

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
main() {
    parse_parameters "$@"

    local experiment_dir
    experiment_dir=$(create_experiment_dir)

    local total_configs=$((${#batch_sizes[@]} * ${#chunk_lengths[@]}))
    echo -e "\n=== Starting Experiment Series ==="
    echo "Model:           $model"
    echo "Input Directory: $input_dir"
    echo "Output Directory: $experiment_dir"
    echo "Total Configurations: $total_configs"
    echo "----------------------------------------"

    local count=1
    for batch_size in "${batch_sizes[@]}"; do
        validate_positive_integer "$batch_size" "Batch size"
        for chunk_len in "${chunk_lengths[@]}"; do
            validate_positive_integer "$chunk_len" "Chunk length"
            echo -e "\n=== Processing Configuration ${count}/${total_configs} ==="
            run_experiment "$batch_size" "$chunk_len" "$experiment_dir"
            ((count++))
        done
    done

    echo -e "\n=== All Experiments Completed ==="
    echo "Final results available in: $experiment_dir"
}

main "$@"