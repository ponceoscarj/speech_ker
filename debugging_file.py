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

error = jiwer.wer(gold_standard, predicted)
print(error)