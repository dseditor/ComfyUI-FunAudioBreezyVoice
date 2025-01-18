'''
Author: SpenserCai
Date: 2024-10-04 12:13:28
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-05 12:23:01
Description: file content
'''
import os
import folder_paths
import numpy as np
import torch
import datetime
from funaudio_utils.pre import FunAudioLLMTool
from funaudio_utils.download_models import download_cosyvoice2_05B,download_cosyvoice_300m, get_speaker_default_path, download_cosyvoice_300m_sft,download_cosyvoice_300m_instruct
from funaudio_utils.cosyvoice_plus import CosyVoice1, CosyVoice2
from cosyvoice.utils.common import set_all_random_seed

fAudioTool = FunAudioLLMTool()

CATEGORY_NAME = "FunAudioLLM_V2/CosyVoice"
SPEAKER_LIST = []

folder_paths.add_model_folder_path("CosyVoice", os.path.join(folder_paths.models_dir, "CosyVoice"))

def return_audio(output,t0,spk_model):
    output_list = []
    for out_dict in output:
        output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
        output_numpy = output_numpy.astype(np.int16)
        # if speed > 1.0 or speed < 1.0:
        #     output_numpy = speed_change(output_numpy,speed,target_sr)
        output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
    t1 = ttime()
    print("cost time \t %.3f" % (t1-t0))
    audio = {"waveform": torch.cat(output_list,dim=1).unsqueeze(0),"sample_rate":fAudioTool.target_sr}
    if spk_model is not None:
        return (audio,spk_model,)
    else:
        return (audio,)

from time import time as ttime


# 零样本音色克隆V2
class CosyVoice2ZeroShotNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            },
            "optional":{
                "prompt_text":("STRING",),
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO","SPK_MODEL",)
    
    FUNCTION="generate"

    def generate(self, tts_text, speed, seed, text_frontend, prompt_text=None, prompt_wav=None, speaker_model=None):
        t0 = ttime()
        _, model_dir = download_cosyvoice2_05B()
        cosyvoice = CosyVoice2(model_dir)
        assert len(tts_text) > 0, "tts_text不能为空！！！"
        if speaker_model is None:
            assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
            speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
            prompt_speech_16k = fAudioTool.postprocess(speech)
            print('get zero_shot inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k,False,speed,text_frontend)
            spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k,24000)
            del spk_model['text']
            del spk_model['text_len']
            return return_audio(output,t0,spk_model)
        else:
            print('get zero_shot inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model,False,speed)
            return return_audio(output,t0,speaker_model)


# 跨语言音色克隆V2    
class CosyVoice2CrossLingualNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            },
            "optional":{
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO","SPK_MODEL",)
    FUNCTION="generate"

    def generate(self, tts_text, speed, seed, text_frontend, prompt_wav=None, speaker_model=None):
        t0 = ttime()
        _, model_dir = download_cosyvoice2_05B()
        cosyvoice = CosyVoice2(model_dir)
        assert len(tts_text) > 0, "tts_text不能为空！！！"
        if speaker_model is None:
            speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
            prompt_speech_16k = fAudioTool.postprocess(speech)
            print('get cross_lingual inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, False, speed,text_frontend)
            spk_model = cosyvoice.frontend.frontend_cross_lingual(tts_text, prompt_speech_16k,24000)
            del spk_model['text']
            del spk_model['text_len']
            return return_audio(output,t0,spk_model)
        else:
            print('get cross_lingual inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model,False,speed)
            return return_audio(output,t0,speaker_model)

# 自然语言控制V2
class CosyVoice2InstructNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "instruct_text":("STRING",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            },
            "optional":{
                "prompt_text":("STRING",),
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO","SPK_MODEL",)
    FUNCTION="generate"

    def generate(self, tts_text, instruct_text, speed, seed, text_frontend,prompt_text=None, prompt_wav=None, speaker_model=None):
        t0 = ttime()
        _, model_dir = download_cosyvoice2_05B()
        cosyvoice = CosyVoice2(model_dir)
        assert len(tts_text) > 0, "tts_text不能为空！！！"
        if speaker_model is None:
            assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
            speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
            prompt_speech_16k = fAudioTool.postprocess(speech)
            print('get instruct2 inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_instruct2(tts_text, instruct_text,prompt_speech_16k, False, speed, text_frontend)
            spk_model = cosyvoice.frontend.frontend_instruct2(tts_text, instruct_text, prompt_speech_16k,24000)
            del spk_model['text']
            del spk_model['text_len']
            return return_audio(output,t0,spk_model)
        else:
            print('get instruct2 inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model,False,speed)
            return return_audio(output,t0,speaker_model)

# 零样本音色克隆
class CosyVoiceZeroShotNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            },
            "optional":{
                "prompt_text":("STRING",),
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO","SPK_MODEL",)
    
    FUNCTION="generate"

    def generate(self, tts_text, speed, seed, use_25hz,text_frontend, prompt_text=None, prompt_wav=None, speaker_model=None):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoice1(model_dir)
        if speaker_model is None:
            assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
            speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
            prompt_speech_16k = fAudioTool.postprocess(speech)
            print('get zero_shot inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k,False,speed,text_frontend)
            spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k,24000)
            del spk_model['text']
            del spk_model['text_len']
            return return_audio(output,t0,spk_model)
        else:
            print('get zero_shot inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model,False,speed)
            return return_audio(output,t0,speaker_model)

# 预训练音色使用
class CosyVoiceSFTNode:
    sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING",  {
                    "default": "",
                    "multiline": True
                }),
                "speaker_name":(s.sft_spk_list,{
                    "default":"中文女"
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, speaker_name, speed, seed, use_25hz, text_frontend):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m_sft(use_25hz)
        cosyvoice = CosyVoice1(model_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, speaker_name, False, speed, text_frontend)
        return return_audio(output,t0,None)

# 跨语言音色克隆    
class CosyVoiceCrossLingualNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "prompt_wav": ("AUDIO",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, prompt_wav, speed, seed, use_25hz, text_frontend):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoice1(model_dir)
        speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
        prompt_speech_16k = fAudioTool.postprocess(speech)
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, False, speed, text_frontend)
        return return_audio(output,t0,None)

# 自然语言控制
class CosyVoiceInstructNode:
    sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "speaker_name":(s.sft_spk_list,{
                    "default":"中文女"
                }),
                "instruct_text":("STRING",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "text_frontend":("BOOLEAN",{
                    "default": True
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, speaker_name, instruct_text, speed, seed, text_frontend):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m_instruct()
        cosyvoice = CosyVoice1(model_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, speaker_name, instruct_text, False, speed, text_frontend)
        return return_audio(output,t0,None)

# 刷新音色列表
def refresh_list():
    file_list = []
    for _, _, files in os.walk(get_speaker_default_path()):
        for file in files:
            if file.endswith('.pt'):
                file_name = os.path.splitext(file)[0]  # 分离文件名和扩展名，并取文件名部分
                file_list.append(file_name)
    return file_list
SPEAKER_LIST = refresh_list()

# 加载音色模型
class CosyVoiceLoadSpeakerModelNode:
    sft_spk_list = SPEAKER_LIST
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required":{
                "speaker_name":(s.sft_spk_list,{
                    "default":"none"
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("SPK_MODEL",)
    FUNCTION="generate"

    def generate(self, speaker_name):
        model_dir = get_speaker_default_path()
        # 加载模型
        spk_model_path = os.path.join(model_dir, speaker_name + ".pt")
        assert os.path.exists(spk_model_path), "Speaker model is not exist"
        spk_model = torch.load(os.path.join(model_dir, speaker_name + ".pt"),map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return (spk_model,)

# 从网络加载音色模型    
class CosyVoiceLoadSpeakerModelFromUrlNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model_url":("STRING", {
                    "default": ""
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("SPK_MODEL",)
    FUNCTION="generate"

    def generate(self, model_url):
        # 下载模型
        spk_model = torch.hub.load_state_dict_from_url(model_url,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return (spk_model,)

# 保存音色模型 
class CosyVoiceSaveSpeakerModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "spk_model":("SPK_MODEL",),
                "speaker_name":("STRING", {
                    "default": "speaker"
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION="generate"

    def generate(self, spk_model, speaker_name):
        # 判断目录是否存在，不存在则创建
        model_dir = get_speaker_default_path()
        for _, _, files in os.walk(model_dir):
            for file in files:
                # 检查文件重名
                if file == speaker_name + '.pt':
                    print('文件名已存在，自动重命名！')
                    now = datetime.datetime.now()
                    speaker_name = speaker_name + f"{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        print(f"saving speaker model {speaker_name} to {model_dir}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # 保存模型
        torch.save(spk_model, os.path.join(model_dir, speaker_name + ".pt"))
        return speaker_name + '.pt'
        
        