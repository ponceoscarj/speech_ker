#!/bin/bash

# ==============================================================================
# Nemo Buffered RNNT Experiment Runner
# ==============================================================================
# This script automates running buffered RNNT inference experiments with various 
# configurations and calculates Word Error Rate (WER).
# ==============================================================================#
# set -eo pipefail  # Enable strict error checking
# trap "echo -e '\nError: Script aborted due to error'; exit 1" ERR

# Example Usage:
# bash run_nemo_rnnt.sh \
#     --model-path /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo \
#     --dataset-manifest /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/work_files/asr_manifest.json \
#     --output-dir /home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/output \
#     --chunk-lengths "20 40 60 80" \
#     --contexts "0 3 5 10 15" \
#     --batch-size "8" \
#     --model-stride 4 \
#     --merge-algo "middle"

# ==============================================================================
# Global Configuration and Defaults
# ==============================================================================
readonly DEFAULT_PRETRAINED=""
readonly DEFAULT_OUTPUT="./results"
readonly DEFAULT_CHUNK_LENS=(20 40 60 80)
readonly DEFAULT_CONTEXTS=(3 5 7 10 15 20 30 40)
readonly DEFAULT_MERGE_ALGO="middle"
readonly DEFAULT_BATCH_SIZE=8
readonly DEFAULT_MODEL_STRIDE=4
readonly DEFAULT_SLEEP=2

# ==============================================================================
# Initialize variables with default values
# ==============================================================================
model_path=""
dataset_manifest=""
pretrained_name="$DEFAULT_PRETRAINED"
output_base_dir="$DEFAULT_OUTPUT"
chunk_lens=("${DEFAULT_CHUNK_LENS[@]}")
contexts=("${DEFAULT_CONTEXTS[@]}")
merge_algo="$DEFAULT_MERGE_ALGO"
batch_size="$DEFAULT_BATCH_SIZE"
model_stride="$DEFAULT_MODEL_STRIDE"
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
  -c, --chunk-lengths LENGTHS        Space-separated chunk lengths in seconds (default: "${DEFAULT_CHUNK_LENS[@]}")
  -x, --contexts CONTEXTS            Space-separated context sizes in seconds (default: "${DEFAULT_CONTEXTS[@]}")
  -a, --merge-algo ALGORITHM         Merge algorithm: 'middle' or 'lcs' (default: "$DEFAULT_MERGE_ALGO")
  -b, --batch-size SIZE              Batch size (default: $DEFAULT_BATCH_SIZE)
  -r, --model-stride STRIDE          Model stride factor (default: $DEFAULT_MODEL_STRIDE)
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
    
    if ! [[ "$value" =~ ^([0-9]+(\.[0-9]*)?|\.[0-9]+)$ ]]; then
        echo "ERROR: ${name} must be a non-negative number (>= 0), got '${value}'"
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
    parsed_args=$(getopt -o m:d:p:o:c:x:a:b:r:s:h \
                --long model-path:,dataset-manifest:,pretrained-name:,output-dir:,chunk-lengths:,contexts:,merge-algo:,batch-size:,model-stride:,sleep-time:,help \
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
                IFS=' ' read -r -a chunk_lens <<< "$trimmed_chunks"
                for chunk in "${chunk_lens[@]}"; do
                    validate_positive_number "${chunk}" "Chunk length"
                done
                shift 2 ;;
            -a|--merge-algo)
                merge_algo="$2"
                if [[ "$merge_algo" != "middle" && "$merge_algo" != "lcs" ]]; then
                    echo "ERROR: Invalid merge algorithm '${merge_algo}'. Must be 'middle' or 'lcs'."
                    exit 1
                fi
                shift 2 ;;                
            -x|--contexts)
                IFS=' ' read -r -a contexts <<< "$(echo "$2" | xargs)"
                for ctx in "${contexts[@]}"; do
                    validate_positive_number "$ctx" "Context"
                done
                shift 2 ;;                
            -b|--batch-size)
                batch_size="$2"
                validate_positive_integer "$batch_size" "Batch size"
                shift 2 ;;
            -r|--model-stride)
                model_stride="$2"
                validate_positive_integer "${model_stride}" "Model stride"
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
    local chunk_len="$2"
    local context="$3"
    local total_buffer=$(("$chunk_len + $context"))
    local batch_size="$4"
    cat <<EOF > "${log_path}"
╔══════════════════════════════════════════════════╗
║            EXPERIMENT PARAMETERS                 ║
╠══════════════════════════════════════════════════╣
  Start Time    │ $(date +'%Y-%m-%d %H:%M:%S %Z')
  ──────────────┼───────────────────────────────────
  Model Path    │ ${model_path}
  Pretrained    │ ${pretrained_name}
  Dataset       │ $(basename "${dataset_manifest}")
  Chunk Length  │ ${chunk_len}s
  Context       │ ${context}s
  Merge Algo    │ ${merge_algo}"
  Total Buffer  │ ${total_buffer}s
  Batch Size    │ ${batch_size}
  Model Stride  │ ${model_stride}
  Sleep Time    │ ${sleep_time}s
╚══════════════════════════════════════════════════╝
EOF
}

# ==============================================================================
# Experiment Execution
# ==============================================================================
run_experiment() {
    local chunk_len="$1"
    local context="$2"
    local experiment_dir="$3"
    local batch_size="$4"
    local total_buffer=$(("$chunk_len + $context"))    
    local model_dir_name=$(basename "${experiment_dir}")
    local timestamp=$(date +'%Y%m%d_%H%M%S')

    # Sanitize float values for filenames
    local chunk_len_sanitized="${chunk_len//./_}"
    local context_sanitized="${context//./_}"    
    
    local log_filename="${model_dir_name}_chunk${chunk_len_sanitized}_ctx${context_sanitized}_batch${batch_size}_mergealgo${merge_algo}.log"
    local log_file="${experiment_dir}/${log_filename}"
    local output_file="${experiment_dir}/${model_dir_name}_chunk${chunk_len_sanitized}_ctx${context_sanitized}_batch${batch_size}_mergealgo${merge_algo}.json"

    echo "Running configuration:"
    echo "  - Buffer Length:    ${total_buffer}s"
    echo "  - Chunk Length:     ${chunk_len}s"
    echo "  - Context Length:   ${context}s"
    echo "  - Merge Algo:       ${merge_algo}"     
    echo "  - Batch Size:   ${batch_size}"
    echo "  - Log file:     ${log_filename}"
    
    # Initialize log with header
    initialize_log_file "${log_file}" "${chunk_len}" "${context}" "${batch_size}" "${merge_algo}"

    {
        echo -e "\033[1;34m┌───────────────────────────────────────────────┐"
        echo -e "│          Starting Transcription Process         │"
        echo -e "└───────────────────────────────────────────────┘\033[0m"
        python3 nemo_buffered_infer_rnnt.py \
            model_path="${model_path}" \
            dataset_manifest="${dataset_manifest}" \
            output_filename="${output_file}" \
            total_buffer_in_secs="${total_buffer}" \
            chunk_len_in_secs="${chunk_len}" \
            model_stride="${model_stride}" \
            merge_algo="${merge_algo}" \
            batch_size="${batch_size}" \
            langid='en' || {
                echo "ERROR: Transcription failed for chunk ${chunk_len}, context ${context}"
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
    echo -e "\n\033[1;36m=== Nemo CTC Experiment Runner ===\033[0m"
    echo -e "\033[1;34m$(date +'%Y-%m-%d %H:%M:%S %Z')\033[0m\n"    
    
    parse_parameters "$@"
    echo "=== Parameters parsed ==="
        
    local experiment_dir=$(create_experiment_dir)
    local total_configs=$((${#chunk_lens[@]} * ${#contexts[@]}))
    
    echo -e "\033[1;35m▬▬▬▬▬▬▬▬▬▬▬▬ Experiment Setup ▬▬▬▬▬▬▬▬▬▬▬▬\033[0m"
    printf "\033[1m%-20s\033[0m %s\n" \
        "Merge Algorithm:" "${merge_algo}" \
        "Model:" "${pretrained_name:-[custom]}" \
        "Dataset:" "$(basename ${dataset_manifest})" \
        "Output Directory:" "${experiment_dir}" \
        "Total Experiments:" "${total_configs}" \
        "Chunk Lengths:" "${chunk_lens[*]} seconds" \
        "Contexts:" "${contexts[*]} seconds" \
        "Batch Size:" "${batch_size}" \
        "Model Stride:" "${model_stride}" \
        "Sleep Between Runs:" "${sleep_time}s"
    echo -e "\033[1;35m▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\033[0m\n"

    local count=1
    local start_time=$SECONDS    

    for chunk in "${chunk_lens[@]}"; do
        for ctx in "${contexts[@]}"; do
            local elapsed=$((SECONDS - start_time))
            local remaining=$(( (total_configs - count + 1) * elapsed / (count) ))
            
            echo -e "\033[1;33m〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰〰\033[0m"
            printf "\033[1;32m[%d/%d] Running Experiment:\033[0m\n" $count $total_configs
            printf "  - Chunk: \033[1m%5.1fs\033[0m\n" $chunk
            printf "  - Context: \033[1m%5.1fs\033[0m\n" $ctx
            local total_buffer
            total_buffer=$(awk "BEGIN { printf \"%.1f\", $chunk + $ctx }")        
            printf "  - Total Buffer: \033[1m%5.1fs\033[0m\n" "$total_buffer"
            printf "  - Batch Size: \033[1m%3d\033[0m\n" $batch_size
            printf "  - Elapsed: %02d:%02d | Est. Remaining: %02d:%02d\n" \
                $((elapsed/60)) $((elapsed%60)) \
                $((remaining/60)) $((remaining%60))
            
            run_experiment "$chunk" "$ctx" "$experiment_dir" "$batch_size"
            ((count++))
        done
    done


    local total_time=$((SECONDS - start_time))
    echo -e "\n\033[1;36m✓✓✓ All Experiments Completed ✓✓✓\033[0m"
    echo -e "\033[1mTotal Duration:\033[0m $((total_time/60))m $((total_time%60))s"
    echo -e "\033[1mResults Directory:\033[0m ${experiment_dir}\n"
}

main "$@"