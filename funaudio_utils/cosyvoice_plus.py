'''
Author: SpenserCai
Date: 2024-10-04 14:21:08
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 16:07:20
Description: file content
'''
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import logging #,load_wav
from tqdm import tqdm
import time
import os

class CosyVoice1(CosyVoice):
    
    def inference_zero_shot_with_spkmodel(self,tts_text, spkmodel,stream=False, speed=1.0):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=True)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(tts_text)
            spkmodel["text"] = tts_text_token
            spkmodel["text_len"] = tts_text_token_len
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**spkmodel, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 24000
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

class CosyVoice2(CosyVoice2):
    
    def inference_zero_shot_with_spkmodel(self,tts_text, spkmodel,stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(tts_text)
            spkmodel["text"] = tts_text_token
            spkmodel["text_len"] = tts_text_token_len
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**spkmodel, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 24000
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

class TextReplacer:
    def __init__(self, input_string, replacement_file=""):
        """
        初始化 TextReplacer 对象并执行替换操作。

        参数:
        input_string (str): 输入的字符串。
        replacement_file (str): 替换规则文件路径。
        """
        self.input_string = input_string
        self.replacement_rules = self.load_replacement_rules_from_txt(replacement_file) if replacement_file else []
        self.result_string = self.replace_phrases()

    def load_replacement_rules_from_txt(self, file_path):
        """
        从 .txt 文件中加载替换规则。

        参数:
        file_path (str): 替换规则文件路径。

        返回:
        list: 替换规则列表，每个规则是一个包含两个元素的元组。
        """
        replacement_rules = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if ',' in line or '，' in line:
                    old, new = line.strip().replace('，', ',').split(',')
                    replacement_rules.append((old, new))
        return replacement_rules

    def replace_phrases(self):
        """
        根据替换规则列表替换字符串中的短语。

        返回:
        str: 替换后的字符串。
        """
        result = self.input_string
        for old, new in self.replacement_rules:
            result = result.replace(old, new)
        return result
    def replace_tts_text(tts_text, file_name="多音字纠正配置.txt"):
        replacement_file= os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        replacer = TextReplacer(tts_text, replacement_file)
        result_string = replacer.result_string
        return result_string