import torch
import nemo.collections.asr as nemo_asr
import contextlib
import gc
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import numpy as np
from torch.utils.data import DataLoader
import math
from nemo.collections.asr.metrics.wer import word_error_rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# Clear up memory
torch.cuda.empty_cache()
gc.collect()
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_conformer_ctc_large", map_location=device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'  # You can transcribe even longer samples on the CPU, though it will take much longer !
model = model.to(device)

# Helper for torch amp autocast
if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        print("AMP was not available, using FP32!")
        yield

concat_audio_path = '/Users/yuqiwu/Documents/PythonProjects/ASR_chunks_2025/audio_files/toy2.wav'

with autocast():
    transcript = model.transcribe([concat_audio_path], batch_size=1)[0]

print(transcript)

# Clear up memory
torch.cuda.empty_cache()
gc.collect()


# A simple iterator class to return successive chunks of samples
class AudioChunkIterator():
    def __init__(self, samples, frame_len, sample_rate):
        self._samples = samples
        self._chunk_len = chunk_len_in_secs * sample_rate
        self._start = 0
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False

        return chunk

# a helper function for extracting samples as a numpy array from the audio file
def get_samples(audio_file, target_sr=16000):
    with sf.SoundFile(audio_file, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read()
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
        samples = samples.transpose()
        return samples

samples = get_samples(concat_audio_path)
sample_rate  = model.preprocessor._cfg['sample_rate']
chunk_len_in_secs = 1
chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
count = 0
for chunk in chunk_reader:
    count +=1
    plt.plot(chunk)
    plt.show()
    if count >= 5:
        break

context_len_in_secs = 1

buffer_len_in_secs = chunk_len_in_secs + 2* context_len_in_secs

buffer_len = sample_rate*buffer_len_in_secs
sampbuffer = np.zeros([buffer_len], dtype=np.float32)

chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
chunk_len = sample_rate*chunk_len_in_secs
count = 0
for chunk in chunk_reader:
    count +=1
    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
    sampbuffer[-chunk_len:] = chunk
    plt.plot(sampbuffer)
    plt.show()
    if count >= 5:
        break

from nemo.core.classes import IterableDataset


def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal = []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths


# simple data layer to pass audio signal
class AudioBuffersDataLayer(IterableDataset):

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return torch.as_tensor(self.signal[self._buf_count - 1], dtype=torch.float32), \
            torch.as_tensor(self.signal_shape[0], dtype=torch.int64)

    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1


class ChunkBufferDecoder:

    def __init__(self, asr_model, stride, chunk_len_in_secs=1, buffer_len_in_secs=3):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_secs
        self.buffer_len = buffer_len_in_secs
        assert (chunk_len_in_secs <= buffer_len_in_secs)

        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_secs = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(self.chunk_len / self.model_stride_in_secs)
        self.blank_id = len(asr_model.decoder.vocabulary)
        self.plot = False

    @torch.no_grad()
    def transcribe_buffers(self, buffers, merge=True, plot=False):
        self.plot = plot
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()
        return self.decode_final(merge)

    def _get_batch_preds(self):

        device = self.asr_model.device
        for batch in iter(self.data_loader):

            audio_signal, audio_signal_len = batch

            audio_signal, audio_signal_len = audio_signal.to(device), audio_signal_len.to(device)
            log_probs, encoded_len, predictions = self.asr_model(input_signal=audio_signal,
                                                                 input_signal_length=audio_signal_len)
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())

    def decode_final(self, merge=True, extra=0):
        self.unmerged = []
        self.toks_unmerged = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len - self.chunk_len) / 2) / self.model_stride_in_secs)

        decoded_frames = []
        all_toks = []
        for pred in self.all_preds:
            ids, toks = self._greedy_decoder(pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)

        for decoded in decoded_frames:
            self.unmerged += decoded[len(decoded) - 1 - delay:len(decoded) - 1 - delay + self.n_tokens_per_chunk]
        if self.plot:
            for i, tok in enumerate(all_toks):
                plt.plot(self.buffers[i])
                plt.show()
                print("\nGreedy labels collected from this buffer")
                print(tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk])
                self.toks_unmerged += tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk]
            print("\nTokens collected from successive buffers before CTC merge")
            print(self.toks_unmerged)

        if not merge:
            return self.unmerged
        return self.greedy_merge(self.unmerged)

    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i])
        return ids, s

    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p.item())
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


chunk_len_in_secs = 4
context_len_in_secs = 2

buffer_len_in_secs = chunk_len_in_secs + 2 * context_len_in_secs

n_buffers = 5

buffer_len = sample_rate * buffer_len_in_secs
sampbuffer = np.zeros([buffer_len], dtype=np.float32)

chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
chunk_len = sample_rate * chunk_len_in_secs
count = 0
buffer_list = []
for chunk in chunk_reader:
    count += 1
    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
    sampbuffer[-chunk_len:] = chunk
    buffer_list.append(np.array(sampbuffer))

    if count >= n_buffers:
        break
stride = 4 # 8 for Citrinet
asr_decoder = ChunkBufferDecoder(model, stride=stride, chunk_len_in_secs=chunk_len_in_secs, buffer_len_in_secs=buffer_len_in_secs )
transcription = asr_decoder.transcribe_buffers(buffer_list, plot=True)

# Final transcription after CTC merge
print(transcription)

# WER calculation
# Collect all buffers from the audio file

ref_transcript = "brings you here today yeah i have this pain in my estk and where is the pain exactly ust al over on the on the left side ok and when di this pain start start just thirty minutes ago ok and just come on randomly or were you doing something strenuous i was just showing the driveway and it came on ok and has that pain been getting worse at all over the last haf  hour it just came on suddenly and it's sorry yeah the pain has been there this whole time and it's gotten worse ever since it started ok and how would you describe the pain it kind of like an aching pain or sharp or tight tightness kind of pain how would you describe i feels ll i feel like there's a lot of pressure on y st and how you rate the pain right now on a scale of  to ten youre being the le someone of pain your in your life t being the wors seven seven ok have you had any similar episodes before no i've never had any chest pain before ok and is the pain just staying in the region left chest area that you mention or is it traveling to any other part of your body no i'm kind of just feeling it right here on the left side ok is there anything that you do that mak the paini get worse or go away or get better i think it's a bit bit worse if i'm moving around or when i was walking in here i think it made it a bit worse but nothing seemed to make it any better since starting ok and does it change at all from changing positions like if you're standing  vers sitting down or laying down i think it's a little bit worse when when i'm laying down ok and other than the pain that you've been having you've been having any other symptoms like cag or difficulty breathing or any pain when you're breathing in ot i've felt a little bit short of breath or having difficulty breathing since yesterday when the pain are sorry since the pain started but just the difficulty breathing ok and have you recently injured your chest or surrounding area at all from fall or anything like that i do play rugby and was tackled by another player yesterday but but my chest felt fine after that ok so but the pain us started hapan hour yeah ok have you have you been traveling at all recently no but at home ok has anyone around you've been sick at all no ok have you been having the symptoms like vomiting or any fevers or chills no nor vomiting but i do feel a little bit hot today ok have you measured your temperature at all i did and it was thirty eight degrees ok and have you been having any kind of swelling in your legs or feet no swelling and my legs ok have you been feeling tired at all like increasingly n know my energy has been good ok have you been having any kind of thumping or poplications or feel like your heart being at all it it does feel like it's beating faster and now usually only feels like this wh i'm playing sports ok and have you noticed any changes in your skin at all any rashes no rashes have you had any call for runny nose or sopre throat andk tho symptoms in the past month a few weeks ago was was all of that went away on its own i haven't had any cough ok and have you been feeling dizzy at all or you fainted no dizzines and no i have fainted at all ok just a few more questions have you had any diagnosis made by physician anything like diabetes or high blood pressure yeah i've been told i have high cholesterol high blood pressure ok and do you take any medications for the the things i do take medications for both blood pressure and cholesterol resum a satin and  st a pro and i take a multi vitamin ok and do you have any allergies to any medications at all no allergies ok have you at all in the past been hospitalized for any reason no spiziations any previous surgery no and within your familys anyone passed away from a heart attack or in cancerone in the family no ok and currently right now do you live alone do you live with someone and where you live like apartment i live in a house with my parents ok and you currently work yeah i drive a bus for the city ok and in your daily routine you say you get enough exercise throughout the week yea usually on sundays i'll go for run ok and how about your diet your diet just regularly usual i feel like it's fairly balanced overall i might eat out a little bit too often but try to eat as many vegetables as i can and  smoke cigarettes at all i do yes been smoking for the last twenty years roughly ok how much do you smoke onnor on an average day but a half a pack to a pack a day ok and do you drink no alcohol ok and any recreational drugs like marijuana no marijuana but i have used crystal math in the past ok and when was the last time that you use crystal six days ago six days ago ok and how often do you use crystal a couple of times a month times a month ok and for how long have you been using crystal for the last seven years seven years ok re thank you"

sampbuffer = np.zeros([buffer_len], dtype=np.float32)

chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, sample_rate)
buffer_list = []
for chunk in chunk_reader:
    sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
    sampbuffer[-chunk_len:] = chunk
    buffer_list.append(np.array(sampbuffer))

asr_decoder = ChunkBufferDecoder(model, stride=stride, chunk_len_in_secs=chunk_len_in_secs, buffer_len_in_secs=buffer_len_in_secs )
transcription = asr_decoder.transcribe_buffers(buffer_list, plot=False)
wer = word_error_rate(hypotheses=[transcription], references=[ref_transcript])

print(f"WER: {round(wer*100,2)}%")