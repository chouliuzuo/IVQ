from module.decoder.models import gvmgen
from module.decoder.data.audio import audio_write
import moviepy.editor as mp
from pydub import AudioSegment
import os
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Script for processing video and model paths.')
    
    parser.add_argument('--model_path', type=str, default='./checkpoint', 
                        help='Path to the model checkpoint.')
    parser.add_argument('--syn_path', type=str, default='./video_v2m',
                        help='Path to the synthesis output directory.')
    parser.add_argument('--fps', type=int, default=1, 
                        help='video sample rate.')
    parser.add_argument('--duration', type=int, default=30, 
                        help='video length.')
    parser.add_argument('--video_pt_dir', type=str, default='./pts-video',
                        help='video pt.')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = gvmgen.GVMGen.get_pretrained(args.model_path, device=device)
    import time
    start_time = time.time()
    
    
    video_files = os.listdir(os.path.join(args.video_pt_dir))
    print(video_files)
    for video_path in video_files:
        model.set_generation_params(top_k=8)

        description = [video_path]

        res = model.generate(descriptions=description)

        for idx, one_wav in enumerate(res):
            
            audio_write(os.path.join(args.syn_path, f"{os.path.splitext(os.path.basename(video_path))[0]}"), one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            
if __name__ == '__main__':
    main()