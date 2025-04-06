import torch
import numpy as np
import soundfile as sf
import librosa
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from nemo.core.classes import IterableDataset
from nemo.collections.asr.parts.utils import streaming_utils
from nemo.collections.asr.metrics.wer import word_error_rate
import contextlib
import gc
import math
import tqdm
import os

# for explanation : https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Buffered_Transducer_Inference.ipynb

# --------------------------
# Configuration and Setup
# --------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
gc.collect()

# --------------------------
# Model Initialization
# --------------------------
model_name = "nvidia/parakeet-rnnt-0.6b"
# model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name).to(device)
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-110m")

# --------------------------
# Audio Configuration
# --------------------------
FIRST_OUTPUT_FOLDER = 'asr_transcription'
OUTPUT_FOLDER = f'{FIRST_OUTPUT_FOLDER}/{model_name}'
INPUT_FOLDER = '/Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files' 
# AUDIO_PATH = '/Users/oscarponce/Documents/PythonProjects/speech_ker/audio_files/toy2.wav'
SAMPLE_RATE = model.preprocessor._cfg['sample_rate']
SUPPORTED_FORMATS = ['.wav', '.flac', '.mp3', '.ogg', '.aac']


# Create output directory but ignore step if exists
os.makedirs(OUTPUT_FOLDER ,exist_ok=True) 

# --------------------------
# Helper Functions/Classes
# --------------------------
@contextlib.contextmanager
def cpu_autocast():
    """Fallback context manager for CPU"""
    print("AMP not available, using FP32!")
    yield

# Set autocast context manager
autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else cpu_autocast

def get_samples(audio_file: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio file"""
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read()
        if f.samplerate != target_sr:
            samples = librosa.core.resample(
                samples, orig_sr=f.samplerate, target_sr=target_sr)
        return samples.transpose()

class AudioChunkIterator:
    """Iterates through audio samples in fixed-length chunks"""
    def __init__(self, samples: np.ndarray, chunk_len_secs: float, sample_rate: int):
        self.samples = samples
        self.chunk_len = int(chunk_len_secs * sample_rate)
        self.start = 0
        self.has_more = True

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if not self.has_more:
            raise StopIteration
            
        end = int(self.start + self.chunk_len)
        if end <= len(self.samples):
            chunk = self.samples[self.start:end]
            self.start = end
        else:
            # Handle last chunk with zero-padding
            chunk = np.zeros(self.chunk_len, dtype='float32')
            remaining = len(self.samples) - self.start
            chunk[:remaining] = self.samples[self.start:len(self.samples)]
            self.has_more = False

        return chunk

#Buffer across the dependent frames of the independent batch of samples
class BatchedFeatureFrameBufferer(streaming_utils.BatchedFeatureFrameBufferer):
    """
    Batched variant of FeatureFrameBufferer where batch dimension is the independent audio samples.
    """

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        super().reset()
        self.limit_frames = [None for _ in range(self.batch_size)]

    def get_batch_frames(self):
        # Exit if all buffers of all samples have been processed
        if all(self.signal_end):
            return []

        # Otherwise sequentially process frames of each sample one by one.
        batch_frames = []
        for idx, frame_reader in enumerate(self.all_frame_reader):

            limit_frames = self.limit_frames[idx]
            try:
                if limit_frames is not None and self.buffer_number >= limit_frames:
                  raise StopIteration()

                frame = next(frame_reader)
                frame = np.copy(frame)

                batch_frames.append(frame)
            except StopIteration:
                # If this sample has finished all of its buffers
                # Set its signal_end flag, and assign it the id of which buffer index
                # did it finish the sample (if not previously set)
                # This will let the alignment module know which sample in the batch finished
                # at which index.
                batch_frames.append(None)
                self.signal_end[idx] = True

                if self.signal_end_index[idx] is None:
                    self.signal_end_index[idx] = self.buffer_number

        self.buffer_number += 1
        return batch_frames

    def set_frame_reader(self, frame_reader, idx, limit_frames=None):
        self.all_frame_reader[idx] = frame_reader
        self.signal_end[idx] = False
        self.signal_end_index[idx] = None
        self.limit_frames[idx] = limit_frames


def inplace_buffer_merge(buffer, data, timesteps, model):
    """
    Merges the new text from the current frame with the previous text contained in the buffer.

    The alignment is based on a Longest Common Subsequence algorithm, with some additional heuristics leveraging
    the notion that the chunk size is >= the context window. In case this assumptio is violated, the results of the merge
    will be incorrect (or at least obtain worse WER overall).
    """
    # If delay timesteps is 0, that means no future context was used. Simply concatenate the buffer with new data.
    if timesteps < 1:
        buffer += data
        return buffer

    # If buffer is empty, simply concatenate the buffer and data.
    if len(buffer) == 0:
        buffer += data
        return buffer

    # Concat data to buffer
    buffer += data
    return buffer


class BatchedFrameASRRNNT(streaming_utils.FrameBatchASR):
    """
    Batched implementation of FrameBatchASR for RNNT models, where the batch dimension is independent audio samples.
    """

    def __init__(self, asr_model, frame_len=1.6, total_buffer=4.0,
        batch_size=32, max_steps_per_timestep: int = 5, stateful_decoding: bool = False):
        '''
        Args:
            asr_model: An RNNT model.
            frame_len: frame's duration, seconds.
            total_buffer: duration of total audio chunk size, in seconds.
            batch_size: Number of independent audio samples to process at each step.
            max_steps_per_timestep: Maximum number of tokens (u) to process per acoustic timestep (t).
            stateful_decoding: Boolean whether to enable stateful decoding for preservation of state across buffers.
        '''
        super().__init__(asr_model, frame_len=frame_len, total_buffer=total_buffer, batch_size=batch_size)

        # OVERRIDES OF THE BASE CLASS
        self.max_steps_per_timestep = max_steps_per_timestep
        self.stateful_decoding = stateful_decoding

        self.all_alignments = [[] for _ in range(self.batch_size)]
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.previous_hypotheses = None
        self.batch_index_map = {
            idx: idx for idx in range(self.batch_size)
        }  # pointer from global batch id : local sub-batch id

        try:
            self.eos_id = self.asr_model.tokenizer.eos_id
        except Exception:
            self.eos_id = -1

        # print("Performing Stateful decoding :", self.stateful_decoding)

        # OVERRIDES
        self.frame_bufferer = BatchedFeatureFrameBufferer(
            asr_model=asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer
        )

        self.reset()

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        super().reset()

        self.all_alignments = [[] for _ in range(self.batch_size)]
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.previous_hypotheses = None
        self.batch_index_map = {idx: idx for idx in range(self.batch_size)}

        self.data_layer = [streaming_utils.AudioBuffersDataLayer() for _ in range(self.batch_size)]
        self.data_loader = [
            DataLoader(self.data_layer[idx], batch_size=1, collate_fn=streaming_utils.speech_collate_fn)
            for idx in range(self.batch_size)
        ]

        self.buffers = []

    def read_audio_file(self, audio_filepath: list, delay, model_stride_in_secs):
        assert len(audio_filepath) == self.batch_size

        # Read in a batch of audio files, one by one
        for idx in range(self.batch_size):
            samples = get_samples(audio_filepath[idx])
            samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
            frame_reader = streaming_utils.AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
            self.set_frame_reader(frame_reader, idx)

    def set_frame_reader(self, frame_reader, idx, limit_frames = None):
        self.frame_bufferer.set_frame_reader(frame_reader, idx, limit_frames)

    @torch.no_grad()
    def infer_logits(self):
        frame_buffers = self.frame_bufferer.get_buffers_batch()

        while len(frame_buffers) > 0:
            # While at least 1 sample has a buffer left to process
            self.frame_buffers += frame_buffers[:]

            for idx, buffer in enumerate(frame_buffers):
                if self.plot:
                  self.buffers.append(buffer[:][0])
                self.data_layer[idx].set_signal(buffer[:])

            self._get_batch_preds()
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self):
        """
        Perform dynamic batch size decoding of frame buffers of all samples.

        Steps:
            -   Load all data loaders of every sample
            -   For all samples, determine if signal has finished.
                -   If so, skip calculation of mel-specs.
                -   If not, compute mel spec and length
            -   Perform Encoder forward over this sub-batch of samples. Maintain the indices of samples that were processed.
            -   If performing stateful decoding, prior to decoder forward, remove the states of samples that were not processed.
            -   Perform Decoder + Joint forward for samples that were processed.
            -   For all output RNNT alignment matrix of the joint do:
                -   If signal has ended previously (this was last buffer of padding), skip alignment
                -   Otherwise, recalculate global index of this sample from the sub-batch index, and preserve alignment.
            -   Same for preds
            -   Update indices of sub-batch with global index map.
            - Redo steps until all samples were processed (sub-batch size == 0).
        """
        device = self.asr_model.device

        data_iters = [iter(data_loader) for data_loader in self.data_loader]

        feat_signals = []
        feat_signal_lens = []

        new_batch_keys = []
        for idx in range(self.batch_size):
            if self.frame_bufferer.signal_end[idx]:
                continue
            try:
                batch = next(data_iters[idx])
            
            except StopIteration:
                continue

            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)

            feat_signals.append(feat_signal)
            feat_signal_lens.append(feat_signal_len)

            # preserve batch indices
            new_batch_keys.append(idx)

        if len(feat_signals) == 0:
            return

        feat_signal = torch.cat(feat_signals, 0)
        feat_signal_len = torch.cat(feat_signal_lens, 0)

        del feat_signals, feat_signal_lens

        encoded, encoded_len = self.asr_model(processed_signal=feat_signal, 
                                            processed_signal_length=feat_signal_len)

        # filter out partial hypotheses from older batch subset
        if self.stateful_decoding and self.previous_hypotheses is not None:
            new_prev_hypothesis = [
                self.previous_hypotheses[self.batch_index_map[global_index_key]]
                for global_index_key in new_batch_keys]            
            self.previous_hypotheses = new_prev_hypothesis

        best_hyp = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=True, 
            partial_hypotheses=self.previous_hypotheses
        )
        valid_hypotheses = []
        for hyp in best_hyp:
            if hyp is not None and len(hyp) > 0:  # Check for None and empty lists
                valid_hypotheses.append(hyp[0])  # Access first valid hypothesis
            else:
                valid_hypotheses.append(None)
            if self.stateful_decoding:
                # preserve last state from hypothesis of new batch indices
                self.previous_hypotheses = best_hyp
        
        for hyp, global_index_key in zip(best_hyp, new_batch_keys):
            if hyp is None:
                continue
            # print('idx', idx)
            # print('hyp', hyp)
            # print('new_batch_keys',new_batch_keys)
            hyp = hyp[0]
            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]


            if not has_signal_ended:
                self.all_alignments[global_index_key].append(hyp.alignments)

            
        preds = []
        for hyp in valid_hypotheses:
            if hyp is not None and hasattr(hyp, 'y_sequence'):
                preds.append(hyp.y_sequence)
            else:
                preds.append(torch.tensor([]))  # Empty tensor for failed cases


        for pred, global_index_key in zip(preds, new_batch_keys):

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_preds[global_index_key].append(pred.cpu().numpy())

        if self.stateful_decoding:
            # State resetting is being done on sub-batch only, global index information is not being updated
            self.previous_hypotheses = valid_hypotheses            
            reset_states = self.asr_model.decoder.initialize_state(encoded)

            for idx, pred in enumerate(preds):
                if len(pred) > 0 and pred[-1] == self.eos_id:
                    # reset states :
                    self.previous_hypotheses[idx].y_sequence = pred[:-1]
                    self.previous_hypotheses[idx].dec_state = self.asr_model.decoder.batch_select_state(
                        reset_states, idx
                    )

        # Position map update
        if len(new_batch_keys) != len(self.batch_index_map):
            for new_batch_idx, global_index_key in enumerate(new_batch_keys):
                self.batch_index_map[global_index_key] = new_batch_idx  # let index point from global pos -> local pos

        del encoded, encoded_len
        del best_hyp, pred

    def transcribe(
        self, tokens_per_chunk: int, delay: int, plot=False,
    ):
        """
        Performs "middle token" alignment prediction using the buffered audio chunk.
        """
        self.plot = plot
        self.infer_logits()

        self.unmerged = [[] for _ in range(self.batch_size)]
        for idx, alignments in enumerate(self.all_alignments):

            signal_end_idx = self.frame_bufferer.signal_end_index[idx]
            if signal_end_idx is None:
                raise ValueError("Signal did not end")

            all_toks = []

            for a_idx, alignment in enumerate(alignments):
                alignment = alignment[len(alignment) - 1 - delay : len(alignment) - 1 - delay + tokens_per_chunk]

                ids, toks = self._alignment_decoder(alignment, self.asr_model.tokenizer, self.blank_id)
                all_toks.append(toks)

                if len(ids) > 0 and a_idx < signal_end_idx:
                    self.unmerged[idx] = inplace_buffer_merge(self.unmerged[idx], ids, delay, model=self.asr_model,)

            if plot:
              for i, tok in enumerate(all_toks):
                #   print("\nGreedy labels collected from this buffer")
                #   print(tok[len(tok) - 1 - delay:len(tok) - 1 - delay + tokens_per_chunk])                
                  self.toks_unmerged += tok[len(tok) - 1 - delay:len(tok) - 1 - delay + tokens_per_chunk]
            #   print("\nTokens collected from successive buffers before RNNT merge")
            #   print(self.toks_unmerged)

        output = []
        for idx in range(self.batch_size):
            output.append(self.greedy_merge(self.unmerged[idx]))
        return output

    def _alignment_decoder(self, alignments, tokenizer, blank_id):
        s = []
        ids = []

        for t in range(len(alignments)):
            for u in range(len(alignments[t])):
                token_id = int(alignments[t][u][1])
                if token_id != blank_id:
                    token = tokenizer.ids_to_tokens([token_id])[0]
                    s.append(token)
                    ids.append(token_id)

                else:
                    # blank token
                    pass

        return ids, s

    def greedy_merge(self, preds):
        decoded_prediction = [p for p in preds]
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


def transcribe_buffers(asr_decoder, samples, num_frames, chunk_len_in_secs, buffer_len_in_secs, model_stride, plot=False):

  model.freeze()
  model_stride_in_secs = asr_decoder.asr_model.cfg.preprocessor.window_stride * model_stride
  tokens_per_chunk = math.ceil(chunk_len_in_secs / model_stride_in_secs)
  mid_delay = math.ceil((chunk_len_in_secs + (buffer_len_in_secs - chunk_len_in_secs) / 2) / model_stride_in_secs)

  batch_size = asr_decoder.batch_size  # Since only one sample buffers are available, batch size = 1

  assert batch_size == 1

  with torch.inference_mode():
    with torch.cuda.amp.autocast():
      asr_decoder.reset()
      asr_decoder.sample_offset = 0

      frame_reader = streaming_utils.AudioFeatureIterator(samples.copy(), asr_decoder.frame_len, asr_decoder.raw_preprocessor, asr_decoder.asr_model.device)
      asr_decoder.set_frame_reader(frame_reader, idx=0, limit_frames=num_frames if num_frames is not None else None)

      transcription = asr_decoder.transcribe(tokens_per_chunk, mid_delay, plot=plot)
  
  return transcription



def process_audio_files(audio_path: str, output_dir: str, chunk_len_in_secs: float = 25.0, 
                        context_len_in_secs: float = 5.0, max_steps_per_timestep: int = 5,
                        stateful_decoding: bool = False):
    try:
        samples = get_samples(audio_path, SAMPLE_RATE)

        buffer_len_in_secs = chunk_len_in_secs + 2* context_len_in_secs
        buffer_len = int(SAMPLE_RATE*buffer_len_in_secs)

        sampbuffer = np.zeros([buffer_len], dtype=np.float32)
        chunk_reader = AudioChunkIterator(samples, chunk_len_in_secs, SAMPLE_RATE)

        chunk_len = int(SAMPLE_RATE*chunk_len_in_secs)
        count = 0
        buffer_list = []
        for chunk in chunk_reader:
            count +=1
            sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
            sampbuffer[-chunk_len:] = chunk
            buffer_list.append(np.array(sampbuffer))


        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            if stateful_decoding:  # Very slow procedure, avoid unless really needed
                decoding_cfg.strategy = "greedy"
            else:
                decoding_cfg.strategy = "greedy_batch"

            decoding_cfg.preserve_alignments = True  # required to compute the middle token for transducers.
            decoding_cfg.fused_batch_size = -1  # temporarily stop fused batch during inference.

        model.change_decoding_strategy(decoding_cfg)

        stride = 4 # 8 for ContextNet
        asr_decoder = BatchedFrameASRRNNT(model, frame_len=chunk_len_in_secs, total_buffer=buffer_len_in_secs, 
                                  batch_size=1, 
                                  max_steps_per_timestep=max_steps_per_timestep, 
                                  stateful_decoding=stateful_decoding)

        samples = get_samples(audio_path)

        n_buffers = None
        transcription = transcribe_buffers(asr_decoder, samples, n_buffers, chunk_len_in_secs, buffer_len_in_secs, stride, plot=True)[0]

        # Save results
        filename = os.path.basename(audio_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        
        with open(output_path, 'w') as f:
            f.write(transcription)

        print(f"\n{'='*40}")
        print(f"FILE: {filename}")
        print(f"{'-'*40}")
        print(f"TRANSCRIPTION:\n{transcription}")
        print(f"{'='*40}\n")

        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")


# --------------------------
# Main Processing
# --------------------------
# Load and preprocess audio
print(f"\nStarting ASR processing for files in: {INPUT_FOLDER}")
processed_files = 0

for filename in os.listdir(INPUT_FOLDER):
    filepath = os.path.join(INPUT_FOLDER, filename)

    if os.path.isfile(filepath) and any(filename.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
        transcribed = process_audio_files(
            audio_path=filepath,
            output_dir=OUTPUT_FOLDER,
            chunk_len_in_secs=20.0,
            context_len_in_secs=5.0,
            max_steps_per_timestep=5,
            stateful_decoding=False
        )
        if transcribed:
            processed_files += 1
    else:
        print(f"Skipping non-audio file: {filename}")

# Final summary
print(f"\nProcessing complete! Successfully processed {processed_files} files\n")
print(f"Transcripts saved to: {os.path.abspath(OUTPUT_FOLDER)}\n")




# --------------------------
# Cleanup
# --------------------------
torch.cuda.empty_cache()
gc.collect()