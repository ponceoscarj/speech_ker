import argparse
from omegaconf import OmegaConf
import glob, os
import wget
import torch
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

'''
Example

config_yml can be: ['diar_infer_general.yaml', 'diar_infer_meeting.yaml', 'diar_infer_telephonic.yaml']

python diarization.py --diarizer_type system_vad \
    --config_yml diar_infer_telephonic.yaml \
    --work_dir /Users/oscarponce/Documents/PythonProjects/speech_ker/asr_work_dir \
    --output_dir  /Users/oscarponce/Documents/PythonProjects/speech_ker/diarization/system_vad \
    --vad_model vad_multilingual_marblenet \
    --speaker_model titanet_large \
    --onset 0.7 \
    --offset 0.5 \
    --pad_offset 0 \
    --manifest_name diarize_manifest \
    --num_workers 0 \
    --ignore_overlap True \
    --min_duration_off 0.15 \
    --min_duration_on 0.1 \
    --pad_onset 0.1 \
    --print_rttm # only if you want to print the RTTM file contents on your terminal

'''

def main():
    parser = argparse.ArgumentParser(description='Run speaker diarization with configurable parameters.')
    parser.add_argument('--diarizer_type', type=str, choices=['neural', 'system_vad'], default='system_vad',
                        help='Type of diarizer: "neural" for MSDD or "system_vad" for clustering (default: system_vad)')
    parser.add_argument('--config_yml', type=str, choices=['diar_infer_general.yaml', 'diar_infer_meeting.yaml', 'diar_infer_telephonic.yaml'], default='diar_infer_telephonic.yaml',
                        help='Config YAML filename (default: diar_infer_telephonic.yaml)')
    # parser.add_argument('--audio_files', type=str, default='./audio_files',
    #                     help='Path to audio files')
    parser.add_argument('--work_dir', type=str, default='./work',
                        help='Working directory where diarization manifest is located')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for results (default: ./outputs)')
    parser.add_argument('--vad_model', type=str, default='vad_multilingual_marblenet',
                        help='Pretrained VAD model name (default: vad_multilingual_marblenet)')
    parser.add_argument('--speaker_model', type=str, default='titanet_large',
                        help='Pretrained speaker embedding model (default: titanet_large)')
    parser.add_argument('--onset', type=float, default=0.8,
                        help='VAD onset threshold (default: 0.8)')
    parser.add_argument('--offset', type=float, default=0.6,
                        help='VAD offset threshold (default: 0.6)')
    parser.add_argument('--pad_offset', type=float, default=0,
                        help='VAD pad offset (default: 0)')
    parser.add_argument('--msdd_model', type=str, default='diar_msdd_telephonic',
                        help='MSDD model for neural diarizer (default: diar_msdd_telephonic)')
    parser.add_argument('--sigmoid_threshold', type=str, default='1.0',
                        help='Comma-separated sigmoid thresholds for MSDD (default: 1.0)')
    parser.add_argument('--oracle_vad', action='store_true',
                        help='Use oracle VAD (default: False)')
    parser.add_argument('--oracle_num_speakers', action='store_true',
                        help='Use oracle number of speakers (default: False)')
    parser.add_argument('--manifest_name', type=str, default='diarize_manifest',
                        help='Manifest filename (without .json) (default: diarize_manifest)')
    # parser.add_argument('--transcript_text', type=str, default='asr_work_dir/asr_outcomes/parakeet_tdt.txt',
    #                     help='Path to transcript text file (default: asr_work_dir/asr_outcomes/parakeet_tdt.txt)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading (default: 0)')
    parser.add_argument('--ignore_overlap', type=bool, default=True,
                        help='Ignore overlap in clustering (default: False)')
    parser.add_argument('--min_duration_off', type=float, default=0.15,
                        help='Removes short silences if the duration is less than the specified minimum duration')
    parser.add_argument('--min_duration_on', type=float, default=0.1,
                        help='Removes short speech segments if the duration is less than the specified minimum duration')
    parser.add_argument('--pad_onset', type=float, default=0.1,
                        help=' Adds the specified duration at the beginning of each speech segment')
    parser.add_argument('--print_rttm', action='store_true',
                        help='Print RTTM file contents (default: False)')
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Download config file if missing
    config_path = os.path.join(args.work_dir, args.config_yml)
    if not os.path.exists(config_path):
        config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{args.config_yml}"
        print(f"Downloading config from {config_url}")
        config_path = wget.download(config_url, args.work_dir)
    
    # Generate manifest file if missing
    manifest_file = os.path.join(args.work_dir, f"{args.manifest_name}.json")
    if not os.path.exists(manifest_file):
        print(f"Manifest file is needed")

    # Load and update config
    config = OmegaConf.load(config_path)

    # Check CUDA
    print("CUDA available:", torch.cuda.is_available())
    # and—if you want to see what device NeMo thinks it’s on:
    print("cfg.device:", getattr(config.diarizer, "device", "<not set>"))

    

    config.num_workers = args.num_workers
    config.diarizer.manifest_filepath = manifest_file
    config.diarizer.out_dir = args.output_dir
    config.diarizer.speaker_embeddings.model_path = args.speaker_model
    config.diarizer.oracle_vad = args.oracle_vad
    config.ignore_overlap = args.ignore_overlap
    config.diarizer.clustering.parameters.oracle_num_speakers = args.oracle_num_speakers

    # Set VAD parameters
    config.diarizer.vad.model_path = args.vad_model
    config.diarizer.vad.parameters.onset = args.onset
    config.diarizer.vad.parameters.offset = args.offset
    config.diarizer.vad.parameters.pad_offset = args.pad_offset
    config.diarizer.vad.parameters.min_duration_off = args.min_duration_off
    config.diarizer.vad.parameters.min_duration_on = args.min_duration_on
    config.diarizer.vad.parameters.pad_onset = args.pad_onset
    

    # Configure MSDD if using neural diarizer
    if args.diarizer_type == 'neural':
        config.diarizer.msdd_model.model_path = args.msdd_model
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [float(t) for t in args.sigmoid_threshold.split(',')]

    # Execute diarization
    if args.diarizer_type == 'system_vad':
        print("Running System VAD Diarizer...")
        diarizer = ClusteringDiarizer(cfg=config)
    else:
        print("Running Neural MSDD Diarizer...")
        diarizer = NeuralDiarizer(cfg=config)
    
    diarizer.diarize()
    
    if args.print_rttm:
        pred_dir = os.path.join(args.output_dir, 'pred_rttms')
        for rttm in glob.glob(os.path.join(pred_dir, '*.rttm')):
            print(f"\nContents of {os.path.basename(rttm)}:")
            with open(rttm, 'r') as f:
                print(f.read())

if __name__ == "__main__":
    main()