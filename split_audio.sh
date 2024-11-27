ROOT=$(git rev-parse --show-toplevel)
AUDIO_FILES="${ROOT}/audio_files"
INDIVIDUAL_FILE_NAME="${ROOT}/original_audio_file/toy_example.wav"
OUTPUT_NAME="${AUDIO_FILES}/1min_second.toy.wav"
ffmpeg -ss 61 -i $INDIVIDUAL_FILE_NAME -to 60 -c copy $OUTPUT_NAME