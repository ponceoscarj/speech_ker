from create_manifest import create_manifest
from overall_variables import DATA_DIR, WORK_DIR, ROOT


output_manifest_name = 'asr_manifest'
create_manifest(data_dir= DATA_DIR, duration=None, work_dir= WORK_DIR, 
                output_file_name=output_manifest_name, type='asr')
