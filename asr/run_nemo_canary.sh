#!/bin/bash

: <<'END_COMMENT'
bash ./run_nemo_canary.sh -d /data/manifest.json \
    -m /models/my_model.nemo \
    -p custom-model \
    -o /custom/output \
    -c "10 20 30" \
    -b "8 16" \
    -s 5 \
    -k 2

bash ./run_nemo_canary.sh --model-path /full_path/my_model.nemo \
            --dataset-manifest /full_path/manifest.json \
            --pretrained-name model-name \
            --output-dir /full_path_to_output/ \
            --chunk-lengths "10 20 30" \
            --batch-sizes "8 16" \
            --beam-size 1 \
            --sleep-time 5 
END_COMMENT

# Help function
usage() {
    echo "Usage: $0 [options]"
    echo "Mandatory Options:"
    echo "  -m, --model-path MODEL_PATH      Path to .nemo model file (must end with .nemo)"
    echo "  -p, --pretrained-name NAME       Pretrained model name (default: nvidia/canary-1b-flash)"
    echo "  -d, --dataset-manifest MANIFEST  Path to .json manifest file (required)"
    echo "  -o, --output-dir OUTPUT_DIR      Base output directory (default: /Users/.../nvidia)"
    echo "  -c, --chunk-lengths LENGTHS      Space-separated chunk lengths (default: 20 30 60 80)"
    echo "  -b, --batch-sizes SIZES          Space-separated batch sizes (default: 4 3 2)"
    echo "  -k, --beam-size SIZE             Decoding beam size (default: 1)"
    echo "Optional:"
    echo "  -s, --sleep-time SECONDS         Sleep between runs (default: 2)"
    echo "  -h, --help                       Show this help message"
    exit 1
}

# Parse arguments with long options
PARSED_ARGS=$(getopt -o m:p:d:o:c:b:s:k:h --long model-path:,pretrained-name:,dataset-manifest:,output-dir:,chunk-lengths:,batch-sizes:,sleep-time:,beam-size:,help -n "$0" -- "$@")

eval set -- "$PARSED_ARGS"

# Initialize variables
model_path=""
pretrained_name=""
dataset_manifest=""
output_base_dir=""
chunk_lengths=()
batch_sizes=()
sleep_time=2
beam_size=""

while true; do
    case "$1" in
        -m|--model-path)
            model_path="$2"
            if [[ "$model_path" != "null" && ! "$model_path" =~ \.nemo$ ]]; then
                echo "Error: Model path must end with .nemo"
                exit 1
            fi
            shift 2 ;;
        -p|--pretrained-name)
            pretrained_name="$2"
            shift 2 ;;
        -d|--dataset-manifest)
            dataset_manifest="$2"
            if [[ ! "$dataset_manifest" =~ \.json$ ]]; then
                echo "Error: Dataset manifest must be a .json file"
                exit 1
            fi
            shift 2 ;;
        -o|--output-dir)
            output_base_dir="$2"
            shift 2 ;;
        -c|--chunk-lengths)
            IFS=' ' read -r -a chunk_lengths <<< "$2"
            shift 2 ;;
        -b|--batch-sizes)
            IFS=' ' read -r -a batch_sizes <<< "$2"
            shift 2 ;;
        -s|--sleep-time)
            sleep_time="$2"
            shift 2 ;;
        -k|--beam-size)
            beam_size="$2"
            shift 2 ;;
        -h|--help)
            usage 
            ;;
        --)
            shift
            break ;;
        *)
            echo "Invalid option: $1"
            exit 1 ;;
    esac
done


# ======== ADD VALIDATION FOR MANDATORY PARAMS ======== 
missing_params=()
[[ -z "$model_path" ]] && missing_params+=("--model-path")
[[ -z "$pretrained_name" ]] && missing_params+=("--pretrained-name")
[[ -z "$dataset_manifest" ]] && missing_params+=("--dataset-manifest")
[[ -z "$output_base_dir" ]] && missing_params+=("--output-dir")
[[ ${#chunk_lengths[@]} -eq 0 ]] && missing_params+=("--chunk-lengths")
[[ ${#batch_sizes[@]} -eq 0 ]] && missing_params+=("--batch-sizes")
[[ -z "$beam_size" ]] && missing_params+=("--beam-size")

if [[ ${#missing_params[@]} -gt 0 ]]; then
    echo "Error: Missing mandatory parameters:"
    printf '  %s\n' "${missing_params[@]}"
    exit 1
fi
# ======== END VALIDATION ======== 


# Create output directory structure
model_dir_name=$(echo "$pretrained_name" | tr '/' '_')
mkdir -p "$output_base_dir/$model_dir_name"

# Main iteration loop
for batch_size in "${batch_sizes[@]}"; do
    for chunk_len in "${chunk_lengths[@]}"; do
        # Generate output filename and log path        
        log_filename="${model_dir_name}_chunk_${chunk_len}_beam_${beam_size}_batch_${batch_size}"
        log_file="$output_base_dir/$model_dir_name/${log_filename}.log"

        # Create parameter header for log
        {
            echo "=== Experiment Parameters ==="
            echo "Start Time: $(date +'%Y-%m-%d %H:%M:%S')"
            echo "Model Path: $model_path"
            echo "Pretrained Name: $pretrained_name"
            echo "Dataset Manifest: $dataset_manifest"
            echo "Chunk Length: $chunk_len"
            echo "Batch Size: $batch_size"
            echo "Beam Size: $beam_size"
            echo "Sleep Time: $sleep_time"
            echo "============================="
        } > "$log_file"  # Overwrite log file for each experiment


        # Generate output filename with parameters
        output_filename="$output_base_dir/$model_dir_name/${pretrained_name}_chunk_${chunk_len}_beam_${beam_size}_batch_${batch_size}.json"
            
        # Run the command and log all output
        {
            echo "Starting transcription at: $(date +'%Y-%m-%d %H:%M:%S')"
            echo "============================="
            # 1. Run inference
            python nemo_to_text_aed_chunked_infer.py \
                model_path="$model_path" \
                pretrained_name="$pretrained_name" \
                dataset_manifest="$dataset_manifest" \
                output_filename="$output_filename" \
                chunk_len_in_secs="$chunk_len.0" \
                batch_size="$batch_size" \
                decoding.beam.beam_size="$beam_size"

            # 2. Calculate WER
            echo -e "\nStarting WER calculation for normalized pred_text"                        
            echo "============================="
            python ./asr/new_wer_calculator.py \
                -i "$output_filename" \
                -v

            echo "============================="
            echo "Experiment completed at: $(date +'%Y-%m-%d %H:%M:%S')"
            echo "Sleeping for ${sleep_time}s at: $(date +'%Y-%m-%d %H:%M:%S')"
            sleep $sleep_time
        } 2>&1 | tee -a "$log_file"  # Capture both stdout and stderr            
        # Sleep between runs
    done
done

echo "All experiments completed successfully!"