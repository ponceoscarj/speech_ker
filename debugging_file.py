import jiwer

gold_standard = "non disabling"
predicted = "nondisabling"

# nondisabling, non-disabling, non disabling

# print(jiwer.RemovePunctuation()(predicted))
# Delete all punctuations. Delete anything in brackets [laughter]. [unintelligible]. - '
# Output, is the output with I'm Im

input
# apostrophes are deleted
# hyphens are deleted

# Need to correct in gold standard
# hyphens should be transferred into 1 space
# consistent numbers

# Check in models output afap 024
# check pre-diabetic, non-disabling
# nondisabling, non disabling, non-disabling


# Decisions
# anti coagulation to anticoagulation

error = jiwer.wer(gold_standard, predicted)
print(error)


from asr.normalizer.normalizer import EnglishTextNormalizer

new_text = f'''pre-excitation syndrome, pre-tibial, pre-syncopal, pre-syncope'''

print('\n',new_text,'\n')
print('\n','normalized_text:\n', EnglishTextNormalizer()(new_text), '\n')