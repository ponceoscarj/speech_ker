import json
import os
from create_manifest import create_manifest
from overall_variables import ROOT, WORK_DIR, DATA_DIR

output_manifest_name = 'aligner_manifest'
if os.path.isfile(f'{WORK_DIR}/{output_manifest_name}.json'): 
    None
else:
    create_manifest(data_dir= DATA_DIR, work_dir= WORK_DIR, duration=None,
    output_file_name=output_manifest_name, type='aligner', 
    url_transcript_text='asr_work_dir/asr_outcomes/parakeet_tdt.txt')

print(os.path.join(WORK_DIR, f'{output_manifest_name}.json'))


