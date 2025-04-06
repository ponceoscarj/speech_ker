from pydub import AudioSegment
t1 = 0 * 1000 #Works in milliseconds
t2 = 100 * 1000
newAudio = AudioSegment.from_wav("audio_files/toy2.wav")
newAudio = newAudio[t1:t2]
newAudio.export('audio_files/toy2.wav', format="wav") #Exports to a wav file in the current path.
