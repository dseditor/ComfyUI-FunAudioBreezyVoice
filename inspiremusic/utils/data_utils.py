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
from torch.utils.data import DataLoader
from inspiremusic.dataset.dataset import Dataset
import numpy as np
import librosa

def audio_process_dataset_and_dataloader(args, configs):
    input_dataset = Dataset(args.input_data, data_pipeline=configs['data_pipeline'], mode='processing', shuffle=True, partition=True)
    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    input_data_loader = DataLoader(input_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    return input_dataset, input_data_loader

def is_silent(wav_path, threshold=0.01, frame_length=2048, hop_length=512):
    y, sr = librosa.load(wav_path, sr=None)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    silent_frames = np.sum(rms < threshold) / len(rms)
    silence_fraction_threshold = 0.95
    return silent_frames >= silence_fraction_threshold

def rich_captions(text=None, tags=None, lyrics=None, chorus="verse", start_time=0.0, end_time=30.0):
    if text is None and tags is None and lyrics is None:
        return None
    else:
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = 30.0
        if chorus is None:
            chorus = "verse"
        captions = f"<|{start_time:.1f}|><|{chorus}|>"
        if tags is not None:
            captions += f"<|{tags}|>"
        if text is not None:
            captions += f"<|{text}|>"
        if lyrics is not None:
            captions += f"<|lyrics|><|{lyrics}|>"
        captions += f"<|{end_time:.1f}|>"
        return captions

def process_tags(infile, outfile, timefile = None):
    key_list = []
    with open(infile, "r") as f:
        for line in f:
            sec = line.strip()
            key_list.append(sec)
    f.close()
    if timefile is None:
        with open(outfile, 'w') as f:
            for k in key_list:
                parts = k.rsplit('_', 1)
                text = parts[0].replace('_', ' ') + ', ' + parts[1]
                caption = rich_captions(text, None, None)
                if caption is not None:
                    f.write("%s\t%s\n" %(k, caption))
        f.close()        
    else:
        times = {}
        with open(timefile, "r") as f:
            for line in f:
                sec = line.strip().split("\t")
                if len(sec) == 2 :
                    times[sec[0]] = sec[1]
        f.close()

        with open(outfile, 'w') as f:
            for k in key_list:
                parts = k.rsplit('_', 1)
                text = parts[0].replace('_', ' ') + ', ' + parts[1]
                if k in times.keys():
                    caption = rich_captions(text, None, None, "verse", 0.0, float(times[k]))
                    if caption is not None:
                        f.write("%s\t%s\n" %(k, caption))
        f.close()

def process_trans(infile, outfile):
    trans = {}
    with open(infile, "r") as f:
        for line in f:
            sec = line.strip().split("\t")
            if len(sec) == 2:
                trans[sec[0]] = sec[1]
            else:
                print(line)
    f.close()
    with open(outfile, 'w') as f:
        for k, v in trans.items():
            f.write("%s\t%s\n" %(k, rich_captions(v)))
    f.close()