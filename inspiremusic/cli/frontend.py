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
from functools import partial
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect

from inspiremusic.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
from inspiremusic.wavtokenizer.decoder.pretrained import WavTokenizer

class InspireMusicFrontEnd:
    def __init__(self,
                 configs: Callable,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 llm_model: str,
                 flow_model: str,
                #  music_tokenizer_ckpt_path: str,
                #  music_token_config_path: str,
                 tokenizer_ckpt_path: str,
                 tokenizer_config_path: str,
                 instruct: bool = False,
                 fast: bool = False,
                 fp16: bool = True,
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.tokenizer_config_path = tokenizer_config_path
        self.tokenizer_ckpt_path = tokenizer_ckpt_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bandwidth_id = torch.tensor([0]).to(self.device)
        self.wavtokenizer = WavTokenizer.from_pretrained_feat(tokenizer_config_path, tokenizer_ckpt_path).to(self.device)

        model = InspireMusicModel(configs['llm'], configs['flow'], configs['hift'], configs['wavtokenizer'], fast, fp16)
        self.model = model.load(llm_model, flow_model, None, tokenizer_ckpt_path + "/wavtokenizer/")

        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_audio_token(self, audio, sample_rate=24000, max_audio_length=30):
        assert audio.shape[1] / sample_rate <= max_audio_length, 'do not support extract audio token for audio longer than 30s'
        audio = torch.tensor(audio, dtype=torch.float32).to(self.device)
        _, audio_token = self.wavtokenizer.encode_infer(audio, bandwidth_id=self.bandwidth_id)
        audio_token = audio_token.squeeze(0).numpy().astype(np.int16) 
        audio_token_len = torch.tensor([audio_token.shape[1]], dtype=torch.int32).to(self.device)
        return audio_token, audio_token_len

    def _extract_audio_feat(self, audio):
        audio_feat = self.feat_extractor(audio).squeeze(dim=0).transpose(0, 1).to(self.device)
        audio_feat = audio_feat.unsqueeze(dim=0)
        audio_feat_len = torch.tensor([audio_feat.shape[1]], dtype=torch.int32).to(self.device)
        return audio_feat, audio_feat_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,]+$', '。', text)
            texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                         token_min_n=60, merge_len=20, comma_split=False))
        else:
            text = spell_out_number(text, self.inflect_parser)
            texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                         token_min_n=60, merge_len=20, comma_split=False))
        if split is False:
            return text
        return texts

    def frontend_text_to_music(self, text, time_start, time_end, chorus):
        text_token, text_token_len = self._extract_text_token(text)
        model_input = {"text": text, "audio_token": None, "audio_token_len": None,
                                "text_token": text_token, "text_token_len": text_token_len,
                                "embeddings": [time_start, time_end, chorus], "raw_text":text}
        return model_input

    def frontend_continuation(self, text, audio, time_start, time_end, chorus, target_sr=24000, max_audio_length=30):
        if text is None:
            text_token = None
            text_token_len = None
        else:
            text_token, text_token_len = self._extract_text_token(text)
        audio_token, audio_token_len = self._extract_audio_token(audio, target_sr, max_audio_length)
        model_input = {"text": text, "audio_token": audio_token, "audio_token_len": audio_token_len,
                                "text_token": text_token, "text_token_len": text_token_len,
                                "embeddings": [time_start, time_end, chorus], "raw_text":text}
        return model_input

    def frontend_zero_shot(self, text, prompt_text, prompt_audio, time_start, time_end, chorus, target_sr=24000):
        text_token, text_token_len = self._extract_text_token(text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        _, sr = torchaudio.load(prompt_audio)
        prompt_audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(prompt_audio)
        audio_feat, audio_feat_len = self._extract_audio_feat(prompt_audio)
        audio_token, audio_token_len = self._extract_audio_token(prompt_audio)

        model_input = {'text': text_token, 'text_len': text_token_len,
                            'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                            'llm_prompt_audio_token': audio_token, 'llm_prompt_audio_token_len': audio_token_len,
                            'flow_prompt_audio_token': audio_token, 'flow_prompt_audio_token_len': audio_token_len,
                            'prompt_audio_feat': audio_feat, 'prompt_audio_feat_len': audio_feat_len,
                            "embeddings": [time_start, time_end, chorus]}

        return model_input
