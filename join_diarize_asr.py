from decimal import Decimal

def join_diarize_asr(url_diarized, url_asr):
    with open(url_diarized,'r') as f:
        text = f.readlines()
        turns = []
        for i in text:
            i = i.rstrip().split()
            turn = [i[7], float(i[3]), float(i[3]) + float(i[4])]
            turns.append(turn)    
        print(turns, '\n')

    with open(url_asr,'r') as f:
        text = f.readlines()
        words = []
        for i in text:
            i = i.rstrip().split()
            word = [i[4], float(i[2]), float(i[2]) + float(i[3])]
            words.append(word)
        print(words, '\n')

    new_turns = []
    for turn in turns:
        transcript = []
        for word in words:
            if word[1]>=turn[1] and word[2]<=turn[2]:
                transcript.append(word[0])
            else:
                continue
        if not transcript:
            continue
        else:        
            turn.append(" ".join(transcript))
            new_turns.append([turn[0], str(Decimal(turn[1]).quantize(Decimal('1e-3'))),
            str(Decimal(turn[2]).quantize(Decimal('1e-3'))), turn[3]])
    
    return new_turns

if __name__ == "__main__":
    # outputs/pred_rttms/toy2.rttm
    # asr_work_dir/nfa_output/ctm/words/toy2.ctm

    # for i in 
    final_transcript = join_diarize_asr(url_diarized='outputs/pred_rttms/toy2.rttm', 
                                        url_asr='asr_work_dir/nfa_output/ctm/words/toy2.ctm')
    print(final_transcript)