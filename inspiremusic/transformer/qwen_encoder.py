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

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inspiremusic.utils.mask import make_pad_mask
from inspiremusic.utils.hinter import hint_once

class QwenEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            pretrain_path: str = "Qwen/Qwen2.0-0.5B",
            trainable: bool = False,
            do_fusion_emb: bool = False,
            fusion_drop_rate: float = 0.0,
    ):
        super(QwenEncoder, self).__init__()
        self.input_size = input_size
        self.trainable = trainable
        self.model = AutoModelForCausalLM.from_pretrained(pretrain_path, device_map="cpu")
        self._output_size = self.model.config.hidden_size
        self.do_fusion_emb = do_fusion_emb
        self.hidden_norm = torch.nn.LayerNorm(self._output_size)
        self.fusion_dropout = nn.Dropout(fusion_drop_rate)
        if do_fusion_emb:
            self.fusion_layer = torch.nn.Linear(self._output_size * 2, self._output_size)
            self.emb_norm = torch.nn.LayerNorm(self._output_size)
            self.fusion_norm = torch.nn.LayerNorm(self._output_size)
            from inspiremusic.transformer.activation import Swish
            self.fusion_act = Swish(self)

        if not self.trainable:
            self.model.eval()

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            input_ids: torch.Tensor,
            ilens: torch.Tensor,
    ):
        device = input_ids.device
        input_ids = torch.clamp(input_ids, min=0, max=None)
        input_masks = (~make_pad_mask(ilens)).to(device).long()
        if not self.trainable:
            with torch.no_grad():
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    output_hidden_states=True
                )
        else:
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=input_masks,
                output_hidden_states=True
            )
        outs = model_outputs.hidden_states[-1]
        outs = self.hidden_norm(outs)
        if self.do_fusion_emb:
            hint_once("fuse embedding and LM outputs", "fuse_emb")
            outs = self.fusion_dropout(self.fusion_act(outs))
            emb = model_outputs.hidden_states[0]
            emb = self.fusion_dropout(self.fusion_act(self.emb_norm(emb)))
            outs = self.fusion_layer(
                torch.cat([outs, emb], dim=-1)
            )
            outs = self.fusion_act(self.fusion_norm(outs))

        return outs, ilens


class QwenEmbeddingEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            pretrain_path: str = "Qwen/Qwen2.0-0.5B",
    ):
        super(QwenEmbeddingEncoder, self).__init__()
        self.input_size = input_size
        from transformers import Qwen2ForCausalLM
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path, torch_dtype=torch.float16, device_map="cpu", attn_implementation="flash_attention_2")
        self._output_size = self.model.config.hidden_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            input_embeds: torch.Tensor,
            ilens: torch.Tensor,
    ):
        input_masks = (~make_pad_mask(ilens)).to(input_embeds.device).long()

        outs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
        )

        return outs.hidden_states[-1], input_masks

    def forward_one_step(self, xs, masks, cache=None):

        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values

        return xs, masks, new_cache


class QwenInputOnlyEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            pretrain_path: str = "Qwen/Qwen2.0-0.5B",
    ):
        super(QwenInputOnlyEncoder, self).__init__()
        self.input_size = input_size
        from transformers import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(pretrain_path, torch_dtype=torch.float16, device_map="cpu", attn_implementation="flash_attention_2")
        self.embed = model.model.embed_tokens
        for p in self.embed.parameters():
            p.requires_grad = False
            # set text embedding to non-trainable

        # self.post_embed = model.model.rotary_emb
        self._output_size = model.config.hidden_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            input_ids: torch.Tensor,
            ilens: torch.Tensor,
    ):
        input_masks = (~make_pad_mask(ilens)).to(input_ids.device).long()

        outs = self.embed(input_ids)

        return outs, input_masks
                                                       