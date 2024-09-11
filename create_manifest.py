import json
import os
# from overall_variables import DATA_DIR, WORK_DIR

def create_manifest(data_dir, duration, work_dir, output_file_name, type, url_transcript_text=None):
    manifest_data = []

    for i, wav_file in enumerate(os.listdir(data_dir)):
        if wav_file.endswith('.wav'):
            wav_file_base = os.path.splitext(wav_file)[0]
            # print(wav_file_base)

            if type == "asr":
                entry = {"audio_filepath": os.path.join(data_dir, wav_file), "duration": None, 
                "taskname":"asr", "source_lang": "en", "target_lang": "en",
                "pnc": "yes", "answer": "na"}

            elif type == "diarization":
                with open(url_transcript_text,'r') as f:
                    text = f.readlines()
                    entry = {"audio_filepath": os.path.join(data_dir, wav_file), "duration": None, 
                    "offset": 0, "label": "infer", "text": text[i], "num_speakers": None}
            
            elif type == 'aligner':
                with open(url_transcript_text,'r') as f:
                    text = f.readlines()
                    entry = {"audio_filepath": os.path.join(data_dir, wav_file), 
                    "text": text[i]}
                
            manifest_data.append(entry)


            # "text": text, "offset": offset, 
            # "duration": duration, "label": label
                #     'num_speakers': None, 
                #     'rttm_filepath': None, 
                #     'uem_filepath' : None                    

    #Write the manifest data to a file
    manifest_new_filepath = os.path.join(work_dir, f'{output_file_name}.json')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    with open(manifest_new_filepath, 'w', encoding='utf-8') as f:
        for i in manifest_data:
            f.write(json.dumps(i))
            f.write('\n')
        # json.dump(manifest_data, f, indent=4)



# if __name__ == "__main__":
    # create_manifest(data_dir=DATA_DIR, text='-', offset=0, duration=None, 
    #                 label='infer', work_dir=WORK_DIR)