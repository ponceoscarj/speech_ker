WORK_DIR=$PWD/asr_work_dir
NEMO_DIR_PATH=$PWD/NeMo
manifest_filepath=$PWD/asr_work_dir/aligner_manifest.json
pretrained_name='nvidia/parakeet-ctc-1.1b'
separator='|'

python $NEMO_DIR_PATH/tools/nemo_forced_aligner/align.py \
  pretrained_name=$pretrained_name \
  manifest_filepath=$manifest_filepath \
  output_dir=$WORK_DIR/nfa_output/ \
  additional_segment_grouping_separator=$separator \
  ass_file_config.vertical_alignment="bottom" \
  ass_file_config.text_already_spoken_rgb=[66,245,212] \
  ass_file_config.text_being_spoken_rgb=[242,222,44] \
  ass_file_config.text_not_yet_spoken_rgb=[223,242,239]