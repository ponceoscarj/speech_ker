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


from asr.normalizer.normalizer import EnglishTextNormalizer

new_text = f'''well, it's doesn't happened to be a positive association. they're definitly more prone to err; such as nodiabetic, non-hypertensive, 
lettin' it happen or letting it happen. nondiabetic is diffferent from non-diabetic and non diabetic?
Afib, thirty-five pounds, 35 pounds, five-years time 5-->??_years time in dollars'''

print(EnglishTextNormalizer()(new_text))