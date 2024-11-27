import nemo.collections.asr as nemo_asr
import os
from create_manifest import create_manifest
from utils import DATA_DIR, WORK_DIR, ROOT
from omegaconf import OmegaConf, open_dict
import nemo.collections.nlp as nemo_nlp
import re

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")

output_manifest_name = 'asr_manifest'


create_manifest(data_dir= DATA_DIR, duration=None, work_dir= WORK_DIR, output_file_name=output_manifest_name, type="asr")



# # print(os.path.join(WORK_DIR, f'{output_manifest_name}.json'))


#obtain all audio files to be transcribed
lst_audio = []
for file in os.listdir(f'{DATA_DIR}'):
    lst_audio.append(os.path.join(DATA_DIR, file))

print(lst_audio)

# # predicted_text = asr_model.transcribe(
# #     audio=f"ask_work_dir/{output_manifest_name}.json",
# #     batch_size=16,  # batch size to run the inference with
# # )


# # load model for punctuation
punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(model_name='punctuation_en_bert')



predicted_text = asr_model.transcribe(
    paths2audio_files=lst_audio,
    batch_size=16)

# print(predicted_text)
# print(predicted_text[0])

# # Add punctuation + save
# with open(f'{WORK_DIR}/asr_outcomes/parakeet_tdt.txt', 'w') as f:
#     for i in predicted_text[0]: 
#         # i = punctuation.add_punctuation_capitalization([i])[0]
#         f.write(i+'\n')


with open('asr_work_dir/asr_outcomes/parakeet_tdt.txt', "r") as f:
    text = f.readlines()
    # print(text)
    create_punct = []
    # align_file = []
    for i in text:
        print(i.rstrip())
        i = i.rstrip()
        i = punctuation.add_punctuation_capitalization([i])
        create_punct.append(i)
        # align_i = i.replace(".", "|")
        # align_file.append(align_i)
    
    with open('asr_work_dir/asr_outcomes/parakeet_tdt_punct.txt', "w") as f:
        for i in create_punct:
            f.write(i[0]+'\n')

    with open('asr_work_dir/asr_outcomes/parakeet_tdt_align.txt', "w") as f:
        for i in create_punct:
            new_line = re.sub(r"\?|\!|\.", "|", i[0])
            f.write(new_line+'\n')


# PUNCTUATION + CAPITALIZATION
# punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(model_name='punctuation_en_bert')








# if hypotheses form a tuple (from RNNT), extract just "best" hypotheses
# if type(hypotheses) == tuple and len(hypotheses) == 2:
#     hypotheses = hypotheses[0]

# timestamp_dict = hypotheses[0].timestep # extract timesteps from hypothesis of first (and only) audio file
# print("Hypothesis contains following timestep information :", list(timestamp_dict.keys()))

# print(predicted_text)
# print(type(predicted_text))

# for i in predicted_text[0]: 
#     print(i)





