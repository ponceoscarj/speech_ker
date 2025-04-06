import torch
import numpy as np
import soundfile as sf
import librosa
import nemo.collections.asr as nemo_asr
from torch.utils.data import DataLoader
from nemo.core.classes import IterableDataset
import contextlib
import gc
import math
import os
import argparse

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='ASR Transcription with Configurable Parameters')
    parser.add_argument('--model', type=str, default='stt_en_conformer_ctc_small',
                      help='ASR model name from NVIDIA NeMo')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for inference')
    parser.add_argument('--output_folder', type=str, default='asr_transcription',
                      help='Root directory for output transcriptions')
    parser.add_argument('--input_folder', type=str, default='./audio_files',
                      help='Directory containing audio files to transcribe')
    parser.add_argument('--chunk_len', type=float, default=25.0,
                      help='Length of audio chunks in seconds')
    parser.add_argument('--context_len', type=float, default=5.0,
                      help='Context window size in seconds')
    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------
    # Configuration and Setup
    # --------------------------
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # --------------------------
    # Model Initialization
    # --------------------------
    try:
        model_name = args.model
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name, 
            map_location=args.device).to(args.device)
    except Exception as e:
        print(f"Error loading model {args.model}: {str(e)}")
        return
    
    # --------------------------
    # Audio Configuration
    # --------------------------
    
    OUTPUT_FOLDER = os.path.join(args.output_folder, args.model)
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
                samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
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
                
            end = self.start + self.chunk_len
            if end <= len(self.samples):
                chunk = self.samples[self.start:end]
                self.start = end
            else:
                # Handle last chunk with zero-padding
                chunk = np.zeros(self.chunk_len, dtype='float32')
                remaining = len(self.samples) - self.start
                chunk[:remaining] = self.samples[self.start:]
                self.has_more = False

            return chunk

    class AudioBuffersDataLayer(IterableDataset):
        """Dataset layer for audio buffer processing"""
        def __init__(self):
            super().__init__()
            self.signal = None
            self.signal_shape = None
            self._count = 0

        def __iter__(self):
            self._count = 0
            return self

        def __next__(self):
            if self._count >= len(self.signal):
                raise StopIteration
            self._count += 1
            return (
                torch.as_tensor(self.signal[self._count - 1]), 
                torch.as_tensor(self.signal_shape[0], dtype=torch.int64))

        def set_signal(self, signals):
            self.signal = signals
            self.signal_shape = self.signal[0].shape

        def __len__(self):
            return len(self.signal)

    def speech_collate_fn(batch):
        """Collate function for audio batches"""
        audio_signal, audio_lengths = zip(*batch)
        max_len = max(audio_lengths).item()
        
        processed = []
        for sig, sig_len in zip(audio_signal, audio_lengths):
            if sig_len < max_len:
                sig = torch.nn.functional.pad(sig, (0, max_len - sig_len.item()))
            processed.append(sig)
        
        return torch.stack(processed), torch.stack(audio_lengths)

    class ChunkBufferDecoder:
        """Handles chunked audio decoding with context"""
        def __init__(self, asr_model, stride: int, chunk_len_secs: float, buffer_len_secs: float):
            self.asr_model = asr_model.eval()
            self.stride = stride
            self.chunk_len_secs = chunk_len_secs
            self.buffer_len_secs = buffer_len_secs
            
            # Setup data pipeline
            self.data_layer = AudioBuffersDataLayer()
            self.data_loader = DataLoader(
                self.data_layer, 
                batch_size=1, 
                collate_fn=speech_collate_fn
            )
            
            # Calculate model parameters
            feature_stride = asr_model._cfg.preprocessor['window_stride']
            self.model_stride_sec = feature_stride * stride
            self.n_tokens_per_chunk = math.ceil(chunk_len_secs / self.model_stride_sec)
            self.blank_id = len(asr_model.decoder.vocabulary)

        @torch.no_grad()
        def transcribe_buffers(self, buffers: list, merge: bool = True) -> str:
            """Process audio buffers through ASR model"""
            self.data_layer.set_signal(buffers)
            self._process_buffers()
            return self._decode_output(merge)

        def _process_buffers(self):
            """Run inference on loaded buffers"""
            self.all_preds = []
            for batch in self.data_loader:
                audio, lengths = [t.to(self.asr_model.device) for t in batch]
                _, _, predictions = self.asr_model(input_signal=audio, input_signal_length=lengths)
                self.all_preds.extend(p.cpu().numpy() for p in torch.unbind(predictions))

        def _decode_output(self, merge: bool) -> str:
            """Decode raw model predictions into text"""
            decoded_frames = []
            delay = math.ceil(
                (self.chunk_len_secs + (self.buffer_len_secs - self.chunk_len_secs)/2) 
                / self.model_stride_sec
            )

            for pred in self.all_preds:
                ids, _ = self._greedy_decoder(pred)
                start_idx = len(ids) - 1 - delay
                decoded_frames += ids[start_idx:start_idx + self.n_tokens_per_chunk]

            return self._merge_tokens(decoded_frames) if merge else decoded_frames

        def _greedy_decoder(self, preds: np.ndarray) -> tuple:
            """Basic greedy decoder implementation"""
            ids = []
            tokens = []
            for idx in preds:
                int_id = int(idx)
                if int_id == self.blank_id:
                    tokens.append("_")
                else:
                    token = self.asr_model.tokenizer.ids_to_tokens([int_id])[0]
                    tokens.append(token)
                ids.append(int_id)
            return ids, tokens

        def _merge_tokens(self, tokens: list) -> str:
            """CTC-like token merging"""
            merged = []
            prev = self.blank_id
            for token in tokens:
                if token != prev and token != self.blank_id:
                    merged.append(token)
                prev = token
            return self.asr_model.tokenizer.ids_to_text(merged)


    def process_audio_file(audio_path: str, output_dir: str, chunk_len: float = 25.0, context_len: float = 5.0):
        """Process a single audio file and save transcription"""
        try:
            # Load and preprocess audio
            samples = get_samples(audio_path, SAMPLE_RATE)
            
            # Configure chunking parameters
            buffer_len = chunk_len + 2 * context_len
            
            # Create buffer iterator
            buffer = np.zeros(int(buffer_len * SAMPLE_RATE), dtype=np.float32)
            chunk_iterator = AudioChunkIterator(samples, chunk_len, SAMPLE_RATE)
            buffers = []

            for chunk in chunk_iterator:
                buffer[:-len(chunk)] = buffer[len(chunk):]
                buffer[-len(chunk):] = chunk
                buffers.append(buffer.copy())

            # Initialize decoder and process
            decoder = ChunkBufferDecoder(
                model,
                stride=4,
                chunk_len_secs=chunk_len,
                buffer_len_secs=buffer_len
            )

            with autocast():
                transcription = decoder.transcribe_buffers(buffers)
            
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
    print(f"\nStarting ASR processing with configuration:")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Chunk length: {args.chunk_len}s")
    print(f"Context length: {args.context_len}s\n")
    
    processed_files = 0

    for filename in os.listdir(args.input_folder):
        filepath = os.path.join(args.input_folder, filename)
        
        if os.path.isfile(filepath) and any(filename.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
            success = process_audio_file(
                audio_path=filepath,
                output_dir=OUTPUT_FOLDER,
                chunk_len=args.chunk_len,
                context_len=args.context_len
            )
            if success:
                processed_files += 1
        else:
            print(f"Skipping non-audio file: {filename}")

    # Final summary
    print(f"\nProcessing complete! Successfully processed {processed_files} files\n")
    print(f"Transcripts saved to: {os.path.abspath(OUTPUT_FOLDER)}\n")

if __name__ == "__main__":
    main()
    # --------------------------
    # Cleanup
    # --------------------------
    torch.cuda.empty_cache()
    gc.collect()

