# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from inspiremusic.utils.mask import make_pad_mask
from inspiremusic.music_tokenizer.vqvae import VQVAE

class MaskedDiff(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 128,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 128, 'sampling_rate': 48000,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 48000},
                generator_model_dir: str = "../../pretrained_models/InspireMusic-Base/music_tokenizer",
                num_codebooks: int = 4
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        self.quantizer = VQVAE( f'{generator_model_dir}/config.json',
                                  f'{generator_model_dir}/model.pt',with_encoder=True).quantizer
        self.quantizer.eval()
        self.num_codebooks  = num_codebooks
        self.cond = None
        self.interpolate = False
                                  
    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:

        audio_token = batch['acoustic_token'].to(device)
        audio_token_len = batch['acoustic_token_len'].to(device)
        audio_token  = audio_token.view(audio_token.size(0),-1,self.num_codebooks)
        if "semantic_token" not in batch:
            token = audio_token[:,:,0]
            token_len = (audio_token_len/self.num_codebooks).long()
    
        else:
            token = batch['semantic_token'].to(device)
            token_len = batch['semantic_token_len'].to(device)

        with torch.no_grad():
            feat = self.quantizer.embed(audio_token)
            feat_len = (audio_token_len/self.num_codebooks).long()

        token = self.input_embedding(token) 
        h, h_lengths = self.encoder(token, token_len)
        h, h_lengths = self.length_regulator(h, feat_len)   

        # get conditions
        if self.cond:
            conds = torch.zeros(feat.shape, device=token.device)
            for i, j in enumerate(feat_len):
                if random.random() < 0.5:
                    continue
                index = random.randint(0, int(0.3 * j))
                conds[i, :index] = feat[i, :index]
            conds = conds.transpose(1, 2)
        else:
            conds = None
        
        mask = (~make_pad_mask(feat_len)).to(h)

        loss, _ = self.decoder.compute_loss(
                feat,
                mask.unsqueeze(1),
                h.transpose(1, 2).contiguous(),
                None,
                cond=conds
        )
            
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  sample_rate):
        assert token.shape[0] == 1

        token = self.input_embedding(torch.clamp(token, min=0)) 
        h, h_lengths = self.encoder(token, token_len)

        if sample_rate == 48000:
            token_len = 2 * token_len

        h, h_lengths = self.length_regulator(h, token_len)  

        # get conditions
        conds = None

        mask = (~make_pad_mask(token_len)).to(h)
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=None,
            cond=conds,
            n_timesteps=10
        )
        return feat