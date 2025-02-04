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

import copy
import os
import re
from typing import Iterable, List, Union
import numpy as np
import torch

from inspiremusic.text.abs_tokenizer import AbsTokenizer
from transformers import AutoTokenizer

def get_tokenizer(tokenizer_name, tokenizer_path):
    if "qwen" in tokenizer_name:
        return QwenTokenizer(tokenizer_path,skip_special_tokens=True)
    else:
        return None

class QwenTokenizer(AbsTokenizer):
    def __init__(
            self,
            token_path: str,
            skip_special_tokens: bool = True,
    ):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
            ]
        }
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def text2tokens(self, line: str) -> List:
        tokens = self.tokenizer([line], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def tokens2text(self, tokens) -> str:
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text



def get_qwen_vocab_size(token_type: str):
    if "qwen1.5" in token_type.lower() or "qwen2.0" in token_type.lower() or "qwen2.5" in token_type.lower():
        # 293 for special and extra tokens, including endoftext, im_start, im_end, endofprompt and others in the future.
        # model.vocab_size = 151936, tokenizer.vocab_size = 151643
        # NOTE: the first three special tokens (endoftext, im_start, im_end) are trained in Chat series models,
        # others are kept in random initialization state.
        return 151643 + 293
    else:
        raise ValueError(f"Unknown tokenizer {token_type}")