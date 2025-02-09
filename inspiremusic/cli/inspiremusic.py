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
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from inspiremusic.cli.frontend import InspireMusicFrontEnd
from inspiremusic.cli.model import InspireMusicModel
from inspiremusic.utils.file_utils import logging
import torch

class InspireMusic:
    def __init__(self, model_dir, load_jit=True, load_onnx=False, fast = False, fp16=True, hub="modelscope"):
        instruct = True if '-Instruct' in model_dir else False

        if model_dir is None:
             model_dir = f"../../pretrained_models/InspireMusic-1.5B-Long"
        if not os.path.isfile(f"{model_dir}/llm.pt"):
            model_name = model_dir.split("/")[-1]
            if hub == "modelscope":
                from modelscope import snapshot_download
                if model_name == "InspireMusic-Base":
                    snapshot_download(f"iic/InspireMusic", local_dir=model_dir)
                else:
                    snapshot_download(f"iic/InspireMusic", local_dir=model_dir)

        with open('{}/inspiremusic.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)

        self.frontend = InspireMusicFrontEnd(configs,
                                          configs['get_tokenizer'],
                                          '{}/llm.pt'.format(model_dir),
                                          '{}/flow.pt'.format(model_dir),
                                          '{}/music_tokenizer/'.format(model_dir),
                                          '{}/wavtokenizer/'.format(model_dir),
                                          instruct,
                                          fast,
                                          fp16,
                                          configs['allowed_special'])

        self.model = InspireMusicModel(configs['llm'], configs['flow'], configs['hift'], configs['wavtokenizer'], fast, fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/music_tokenizer/'.format(model_dir),
                        '{}/wavtokenizer/model.pt'.format(model_dir))
        del configs

    @torch.inference_mode()
    def inference(self, task, text, audio, time_start, time_end, chorus, stream=False, sr=24000):
        if task == "text-to-music":
            for i in tqdm(self.frontend.text_normalize(text, split=True)):
                model_input = self.frontend.frontend_text_to_music(i, time_start, time_end, chorus)
                start_time = time.time()
                logging.info('prompt text {}'.format(i))
                for model_output in self.model.inference(**model_input, stream=stream):
                    music_audios_len = model_output['music_audio'].shape[1] / sr
                    logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
                    yield model_output
                    start_time = time.time()
                    
        elif task == "continuation":
            if text is None:
                if audio is not None:
                    for i in tqdm(audio):
                        model_input = self.frontend.frontend_continuation(None, i, time_start, time_end, chorus, sr, max_audio_length)
                        start_time = time.time()
                        logging.info('prompt text {}'.format(i))
                        for model_output in self.model.continuation_inference(**model_input, stream=stream):
                            music_audios_len = model_output['music_audio'].shape[1] / sr
                            logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
                            yield model_output
                            start_time = time.time()
            else:
                if audio is not None:
                    for i in tqdm(self.frontend.text_normalize(text, split=True)):
                        model_input = self.frontend.frontend_continuation(i, audio, time_start, time_end, chorus, sr, max_audio_length)
                        start_time = time.time()
                        logging.info('prompt text {}'.format(i))
                        for model_output in self.model.continuation_inference(**model_input, stream=stream):
                            music_audios_len = model_output['music_audio'].shape[1] / sr
                            logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
                            yield model_output
                            start_time = time.time()
                else:
                    print("Please input text or audio.")
        else:
            print("Currently only support text-to-music and music continuation tasks.")

    @torch.inference_mode()
    def cli_inference(self, text, audio_prompt,sample_rate, time_start, time_end, chorus, task, stream=False, duration_to_gen=30, sr=24000):
        if task == "text-to-music":
            model_input = self.frontend.frontend_text_to_music(text, time_start, time_end, chorus)
            logging.info('prompt text {}'.format(text))
        elif task == "continuation":
            model_input = self.frontend.frontend_continuation(text, audio_prompt, time_start, time_end, chorus, sr)
            logging.info('prompt audio length: {}'.format(len(audio_prompt)))

        start_time = time.time()
        for model_output in self.model.inference(**model_input, duration_to_gen=duration_to_gen, task=task):
            music_audios_len = model_output['music_audio'].shape[1] / sr
            logging.info('yield music len {}, rtf {}'.format(music_audios_len, (time.time() - start_time) / music_audios_len))
            yield model_output
            start_time = time.time()

    @torch.inference_mode()
    def inference_zero_shot(self, text, prompt_text, prompt_audio_16k, stream=False, sr=24000):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(text, split=True)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_audio_16k)
            start_time = time.time()
            logging.info('prompt text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                audio_len = model_output['music_audio'].shape[1] / sr
                logging.info('yield audio len {}, rtf {}'.format(audio_len, (time.time() - start_time) / audio_len))
                yield model_output
                start_time = time.time()
    @torch.inference_mode()
    def inference_instruct(self, text, spk_id, instruct_text, stream=False, sr=24000):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        for i in tqdm(self.frontend.text_normalize(text, split=True)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('prompt text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                audio_len = model_output['music_audio'].shape[1] / sr
                logging.info('yield audio len {}, rtf {}'.format(audio_len, (time.time() - start_time) / audio_len))
                yield model_output
                start_time = time.time()
