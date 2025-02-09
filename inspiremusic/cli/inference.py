# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torchaudio
import time
import logging
import argparse
from inspiremusic.cli.inspiremusic import InspireMusic
from inspiremusic.utils.file_utils import logging
import torch
from inspiremusic.utils.audio_utils import trim_audio, fade_out, process_audio

def set_env_variables():
    os.environ['PYTHONIOENCODING'] = 'UTF-8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    current_working_dir = os.getcwd()
    main_root = os.path.realpath(os.path.join(current_working_dir, '../../'))
    bin_dir = os.path.join(main_root, 'inspiremusic')
    third_party_matcha_tts_path = os.path.join(main_root, 'third_party', 'Matcha-TTS')
    python_path = f"{main_root}:{bin_dir}:{third_party_matcha_tts_path}:{os.environ.get('PYTHONPATH', '')}"
    os.environ['PYTHONPATH'] = python_path
    sys.path.extend([main_root, third_party_matcha_tts_path])

class InspireMusicUnified:
    def __init__(self,
                 model_name: str = "InspireMusic-1.5B-Long",
                 model_dir: str = None,
                 min_generate_audio_seconds: float = 10.0,
                 max_generate_audio_seconds: float = 30.0,
                 sample_rate: int = 24000,
                 output_sample_rate: int = 48000,
                 load_jit: bool = True,
                 load_onnx: bool = False,
                 fast: bool = False,
                 fp16: bool = True,
                 gpu: int = 0,
                 result_dir: str = None,
                 hub="modelscope"):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

        # Set model_dir or default to downloading if it doesn't exist
        if model_dir is None:
             model_dir = f"../../pretrained_models/{model_name}"

        if not os.path.isfile(f"{model_dir}/llm.pt"):
            if hub == "modelscope":
                from modelscope import snapshot_download
                if model_name == "InspireMusic-Base":
                    snapshot_download(f"iic/InspireMusic", local_dir=model_dir)
                else:
                    snapshot_download(f"iic/{model_name}", local_dir=model_dir)

        self.model_dir = model_dir

        self.sample_rate = sample_rate
        self.output_sample_rate = 24000 if fast else output_sample_rate
        self.result_dir = result_dir or f"exp/{model_name}"
        os.makedirs(self.result_dir, exist_ok=True)

        self.min_generate_audio_seconds = min_generate_audio_seconds
        self.max_generate_audio_seconds = max_generate_audio_seconds
        self.min_generate_audio_length = int(self.output_sample_rate * self.min_generate_audio_seconds)
        self.max_generate_audio_length = int(self.output_sample_rate * self.max_generate_audio_seconds)
        assert self.min_generate_audio_seconds <= self.max_generate_audio_seconds, "Min audio seconds must be less than or equal to max audio seconds"

        use_cuda = gpu >= 0 and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = InspireMusic(self.model_dir, load_jit=load_jit, load_onnx=load_onnx, fast=fast, fp16=fp16)
        self.model.model.llm = self.model.model.llm.to(torch.float16)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @torch.inference_mode()
    def inference(self,
                  task: str = 'text-to-music',
                  text: str = None,
                  audio_prompt: torch.Tensor = None, # audio prompt file path
                  sample_rate: int =24000,
                  instruct: str = None,
                  chorus: str = "verse",
                  time_start: float = 0.0,
                  time_end: float = 30.0,
                  output_fn: str = "output_audio",
                  max_audio_prompt_length: float = 5.0,
                  fade_out_duration: float = 1.0,
                  output_format: str = "wav",
                  fade_out_mode: bool = True,
                  trim: bool = False,
                  ):

        with torch.no_grad():
            text_prompt = f"<|{time_start}|><|{chorus}|><|{text}|><|{time_end}|>"
            chorus_dict = {"random": torch.randint(1, 5, (1,)).item(), "intro" : 0, "verse": 1, "chorus": 2, "outro": 4}
            chorus = chorus_dict.get(chorus, 1)
            chorus = torch.tensor([chorus], dtype=torch.int).to(self.device)

            time_start_tensor = torch.tensor([time_start], dtype=torch.float64).to(self.device)
            time_end_tensor = torch.tensor([time_end], dtype=torch.float64).to(self.device)

            music_fn = os.path.join(self.result_dir, f'{output_fn}.{output_format}')

            bench_start = time.time()

            if task == 'text-to-music':
                model_input = {
                    "text"           : text,
                    "audio_prompt"   : audio_prompt,
                    "sample_rate"    : sample_rate,
                    "time_start"     : time_start_tensor,
                    "time_end"       : time_end_tensor,
                    "chorus"         : chorus,
                    "task"           : task,
                    "stream"         : False,
                    "duration_to_gen": self.max_generate_audio_seconds,
                    "sr"             : self.sample_rate
                }
            elif task == 'continuation':
                if audio_prompt is not None:
                    audio, _ = process_audio(audio_prompt,sample_rate ,self.sample_rate)
                    if audio.size(1) < self.sample_rate:
                        logging.warning("Warning: Input prompt audio length is shorter than 1s. Please provide an appropriate length audio prompt and try again.")
                        audio = None
                    else:
                        max_audio_prompt_length_samples = int(max_audio_prompt_length * self.sample_rate)
                        audio = audio[:, :max_audio_prompt_length_samples]  # Trimming prompt audio

                model_input = {
                    "text"           : text,
                    "audio_prompt"   : audio,
                    "sample_rate"    : sample_rate,
                    "time_start"     : time_start_tensor,
                    "time_end"       : time_end_tensor,
                    "chorus"         : chorus,
                    "task"           : task,
                    "stream"         : False,
                    "duration_to_gen": self.max_generate_audio_seconds,
                    "sr"             : self.sample_rate
                }

            music_audios = []
            for model_output in self.model.cli_inference(**model_input):
                music_audios.append(model_output['music_audio'])

            bench_end = time.time()

            if trim:
                music_audio = trim_audio(music_audios[0],
                                         sample_rate=self.output_sample_rate,
                                         threshold=0.05,
                                         min_silence_duration=0.8)
            else:
                music_audio = music_audios[0]

            if music_audio.shape[0] != 0:
                if music_audio.shape[1] > self.max_generate_audio_length:
                    music_audio = music_audio[:, :self.max_generate_audio_length]

                if music_audio.shape[1] >= self.min_generate_audio_length:
                    try:
                        if fade_out_mode:
                            music_audio = fade_out(music_audio, self.output_sample_rate, fade_out_duration)

                        music_audio = music_audio.repeat(2, 1)

                        # if output_format in ["wav", "flac"]:
                        #     torchaudio.save(music_fn, music_audio,
                        #                     sample_rate=self.output_sample_rate,
                        #                     encoding="PCM_S",
                        #                     bits_per_sample=24)
                        # elif output_format in ["mp3", "m4a"]:
                        #     torchaudio.backend.sox_io_backend.save(
                        #         filepath=music_fn, src=music_audio,
                        #         sample_rate=self.output_sample_rate,
                        #         format=output_format)
                        # else:
                        #     logging.info("Format is not supported. Please choose from wav, mp3, m4a, flac.")

                    except Exception as e:
                        logging.error(f"Error saving file: {e}")
                        raise

                audio_duration = music_audio.shape[1] / self.output_sample_rate
                rtf = (bench_end - bench_start) / audio_duration
                logging.info(f"Processing time: {int(bench_end - bench_start)}s, audio length: {int(audio_duration)}s, rtf: {rtf}, text prompt: {text_prompt}")

            else:
                logging.error(f"Generated audio length is shorter than minimum required audio length.")
        return music_audio

def get_args():
    parser = argparse.ArgumentParser(description='Run inference with your model')
    parser.add_argument('-m', '--model_name', default="InspireMusic-1.5B-Long",
                        help='Model name')
    
    parser.add_argument('-d', '--model_dir',
                        help='Model folder path')
    
    parser.add_argument('-t', '--text', default="Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance.",
                        help='Prompt text')
    
    parser.add_argument('-a', '--audio_prompt', default=None, 
                        help='Prompt audio')
    
    parser.add_argument('-c', '--chorus', default="intro", 
                        help='Chorus tag generation mode (e.g., random, verse, chorus, intro, outro)')
    
    parser.add_argument('-f', '--fast', type=bool, default=False, 
                        help='Enable fast inference mode (without flow matching)')

    parser.add_argument('-g', '--gpu', type=int, default=0, 
                        help='GPU ID for this rank, -1 for CPU')
    
    parser.add_argument('--task', default='text-to-music', choices=['text-to-music', 'continuation', 'reconstruct', 'super_resolution'],
                        help='Inference task type: text-to-music, continuation, reconstruct, super_resolution')
    
    parser.add_argument('-r', '--result_dir', default="exp/inspiremusic", 
                        help='Directory to save generated audio')
    
    parser.add_argument('-o', '--output_fn', default="output_audio", 
                        help='Output file name')
    
    parser.add_argument('--format', type=str, default="wav", choices=["wav", "mp3", "m4a", "flac"],
                        help='Format of output audio')
    
    parser.add_argument('--sample_rate', type=int, default=24000, 
                        help='Sampling rate of input audio')
    
    parser.add_argument('--output_sample_rate', type=int, default=48000, choices=[24000, 48000],
                        help='Sampling rate of generated output audio')
    
    parser.add_argument('-s', '--time_start', type=float, default=0.0, 
                        help='Start time in seconds')
    
    parser.add_argument('-e', '--time_end', type=float, default=30.0, 
                        help='End time in seconds')
    
    parser.add_argument('--max_audio_prompt_length', type=float, default=5.0, 
                        help='Maximum audio prompt length in seconds')
    
    parser.add_argument('--min_generate_audio_seconds', type=float, default=10.0, 
                        help='Minimum generated audio length in seconds')
    
    parser.add_argument('--max_generate_audio_seconds', type=float, default=30.0, 
                        help='Maximum generated audio length in seconds')

    parser.add_argument('--fp16', type=bool, default=True, 
                        help='Inference with fp16 model')
    
    parser.add_argument('--fade_out', type=bool, default=True, 
                        help='Apply fade out effect to generated audio')
    
    parser.add_argument('--fade_out_duration', type=float, default=1.0, 
                        help='Fade out duration in seconds')
    
    parser.add_argument('--trim', type=bool, default=False, 
                        help='Trim the silence ending of generated audio')    

    args = parser.parse_args()

    if not args.model_dir:
        args.model_dir = os.path.join("../../pretrained_models", args.model_name)

    print(args)
    return args

def main():
    set_env_variables()
    args = get_args()
    model = InspireMusicUnified(model_name = args.model_name,
                 model_dir = args.model_dir,
                 min_generate_audio_seconds = args.min_generate_audio_seconds,
                 max_generate_audio_seconds = args.max_generate_audio_seconds,
                 sample_rate = args.sample_rate,
                 output_sample_rate = args.output_sample_rate,
                 load_jit = True,
                 load_onnx = False,
                 fast = args.fast,
                 fp16 = args.fp16,
                 gpu = args.gpu,
                 result_dir = args.result_dir)

    model.inference(task = args.task,
                text = args.text,
                audio_prompt = args.audio_prompt,
                chorus = args.chorus,
                time_start = args.time_start,
                time_end = args.time_end,
                output_fn = args.output_fn,
                max_audio_prompt_length = args.max_audio_prompt_length,
                fade_out_duration = args.fade_out_duration,
                output_format = args.format,
                fade_out_mode = args.fade_out,
                trim = args.trim)

if __name__ == "__main__":
    main()