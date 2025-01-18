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

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import torch
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from inspiremusic.cli.model import InspireMusicModel
from inspiremusic.dataset.dataset import Dataset
from inspiremusic.utils.common import MUSIC_STRUCTURE_LABELS

def get_args():
    parser = argparse.ArgumentParser(description='inference only with flow model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--prompt_data', required=True, help='prompt data file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--llm_model', default=None,required=False, help='llm model file')

    parser.add_argument('--music_tokenizer', required=True, help='music tokenizer model file')
    parser.add_argument('--wavtokenizer', required=True, help='wavtokenizer model file')
    parser.add_argument('--chorus', default="random",required=False, help='chorus tag generation mode, eg. random, verse, chorus, intro.')
    parser.add_argument('--sample_rate', type=int, default=48000, required=False,
                        help='sampling rate of generated audio')
    parser.add_argument('--min_generate_audio_seconds', type=float, default=10.0, required=False,
                        help='the minimum generated audio length in seconds')
    parser.add_argument('--max_generate_audio_seconds', type=float, default=30.0, required=False,
                        help='the maximum generated audio length in seconds')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init inspiremusic models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)

    model = InspireMusicModel(None, configs['flow'], configs['hift'], configs['wavtokenizer'])
    model.load(args.llm_model, args.flow_model, args.music_tokenizer, args.wavtokenizer)

    if args.llm_model is None:
        model.llm = None
    else:
        model.llm = model.llm.to(torch.float32)

    if args.flow_model is None:
        model.flow = None

    test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=True, partition=False)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    del configs
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, 'wav.scp')
    f = open(fn, 'w')
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_data_loader)):
            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only support batchsize 1"

            if "semantic_token" in batch:                      
                token  = batch["semantic_token"].to(device)  
                token_len  = batch["semantic_token_len"].to(device)   
            else:
                if audio_token is None:  
                    token = None
                    token_len = None
                else:                                                            
                    token = audio_token.view(audio_token.size(0),-1,4)[:,:,0]
                    token_len  = audio_token_len / 4 

            text_token = batch["text_token"].to(device)
            text_token_len = batch["text_token_len"].to(device)
            text = batch["text"]

            if "time_start" not in batch.keys():
                batch["time_start"] = torch.randint(0, args.min_generate_audio_seconds, (1,)).to(torch.float64)
            if "time_end" not in batch.keys():
                batch["time_end"] = torch.randint(args.min_generate_audio_seconds, args.max_generate_audio_seconds, (1,)).to(torch.float64)
            elif (batch["time_end"].numpy()[0] - batch["time_start"].numpy()[0]) < args.min_generate_audio_seconds:
                batch["time_end"] = torch.randint(int(batch["time_start"].numpy()[0] + args.min_generate_audio_seconds), int(batch["time_start"].numpy()[0] + args.max_generate_audio_seconds), (1,)).to(torch.float64)

            if "chorus" not in batch.keys():
                batch["chorus"] = torch.randint(1, 5, (1,))
            
            if args.chorus == "random":
                batch["chorus"] = torch.randint(1, 5, (1,))
            elif args.chorus == "intro":
                batch["chorus"] = torch.Tensor([0])
            elif "verse" in args.chorus:
                batch["chorus"] = torch.Tensor([1])
            elif args.chorus == "chorus":
                batch["chorus"] = torch.Tensor([2])
            elif args.chorus == "outro":
                batch["chorus"] = torch.Tensor([4])

            time_start = batch["time_start"].to(device)
            time_end = batch["time_end"].to(device)
            chorus = batch["chorus"].to(torch.int)

            text_prompt = f"<|{batch['time_start'].numpy()[0]}|><|{MUSIC_STRUCTURE_LABELS[chorus.numpy()[0]]}|><|{batch['text'][0]}|><|{batch['time_end'].numpy()[0]}|>"
            chorus = chorus.to(device)

            model_input = {"text": text, "audio_token": token, "audio_token_len": token_len,
                                "text_token": text_token, "text_token_len": text_token_len,
                                "embeddings": [time_start, time_end, chorus], "raw_text":text}
                             
            music_audios = []
            for model_output in model.inference(**model_input):
                music_audios.append(model_output['music_audio'])

            music_key = utts[0]
            music_fn = os.path.join(args.result_dir, '{}.wav'.format(music_key))
            torchaudio.save(music_fn, music_audios[0], sample_rate=args.sample_rate)
            f.write('{} {}\n'.format(music_key, music_fn))
            f.flush()
    f.close()
    logging.info('Result wav.scp saved in {}'.format(fn))

if __name__ == '__main__':
    main()