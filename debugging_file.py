import jiwer

reference = "decision making"
hypothesis = "decision-making"

print(jiwer.RemovePunctuation()(hypothesis))

error = jiwer.wer(reference, hypothesis)
print(error)