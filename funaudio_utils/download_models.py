'''
Author: SpenserCai
Date: 2024-10-04 13:54:01
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 22:26:22
Description: file content
'''
import modelscope
import os
import folder_paths
from modelscope import snapshot_download

# Download the model
base_cosyvoice_model_path = os.path.join(folder_paths.models_dir, "CosyVoice")
base_sensevoice_model_path = os.path.join(folder_paths.models_dir, "SenseVoice")
base_InspireMusic_model_path = os.path.join(folder_paths.models_dir, "InspireMusic")

def download_cosyvoice2_05B(auto_download = False):
    model_name = "CosyVoice2-0.5B"
    model_id = "iic/CosyVoice2-0.5B"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_cosyvoice_300m(is_25hz=False,auto_download = False):
    model_name = "CosyVoice-300M"
    model_id = "iic/CosyVoice-300M"
    if is_25hz:
        model_name = "CosyVoice-300M-25Hz"
        model_id = "iic/CosyVoice-300M-25Hz"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_cosyvoice_300m_sft(is_25hz=False,auto_download = False):
    model_name = "CosyVoice-300M-SFT"
    model_id = "iic/CosyVoice-300M-SFT"
    if is_25hz:
        model_name = "CosyVoice-300M-SFT-25Hz"
        model_id = "MachineS/CosyVoice-300M-SFT-25Hz"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_sensevoice_small(auto_download = False):
    model_name = "SenseVoiceSmall"
    model_id = "iic/SenseVoiceSmall"
    model_dir = os.path.join(base_sensevoice_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_cosyvoice_300m_instruct(auto_download = False):
    model_name = "CosyVoice-300M-Instruct"
    model_id = "iic/CosyVoice-300M-Instruct"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_inspiremusic_base_24k(auto_download = False):
    model_name = "InspireMusic-Base-24kHz"
    model_id = "iic/InspireMusic-Base-24kHz"
    model_dir = os.path.join(base_InspireMusic_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    replace_text_in_yaml_files(model_dir,base_InspireMusic_model_path,"InspireMusic-Base-24kHz")
    return model_name, model_dir
def download_inspiremusic_base_48k(auto_download = False):
    model_name = "InspireMusic"
    model_id = "iic/InspireMusic"
    model_dir = os.path.join(base_InspireMusic_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir,)
    replace_text_in_yaml_files(model_dir,base_InspireMusic_model_path,"InspireMusic")
    return model_name, model_dir
def download_inspiremusic_1dot5B_24k(auto_download = False):
    model_name = "InspireMusic-1.5B-24kHz"
    model_id = "iic/InspireMusic-1.5B-24kHz"
    model_dir = os.path.join(base_InspireMusic_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    replace_text_in_yaml_files(model_dir,base_InspireMusic_model_path)
    return model_name, model_dir
def download_inspiremusic_1dot5B_48k(auto_download = False):
    model_name = "InspireMusic-1.5B"
    model_id = "iic/InspireMusic-1.5B"
    model_dir = os.path.join(base_InspireMusic_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    replace_text_in_yaml_files(model_dir,base_InspireMusic_model_path)
    return model_name, model_dir
def download_inspiremusic_1dot5B_long(auto_download = False):
    model_name = "InspireMusic-1.5B-Long"
    model_id = "iic/InspireMusic-1.5B-Long"
    model_dir = os.path.join(base_InspireMusic_model_path, model_name)
    if auto_download:
        snapshot_download(model_id=model_id, local_dir=model_dir)
    replace_text_in_yaml_files(model_dir,base_InspireMusic_model_path)
    return model_name, model_dir

def get_speaker_default_path():
    return os.path.join(base_cosyvoice_model_path, "Speaker")

import os

def replace_text_in_yaml_files(folder_path,replace_text,debug = None):
    # 遍历指定文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为 inspiremusic.yaml
            if file == 'inspiremusic.yaml':
                file_path = os.path.join(root, file)
                try:
                    # 打开文件并读取内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if '../../pretrained_models' in content:
                        # 替换指定文本
                        replace_text = replace_text.replace('\\', '/')
                        new_content = content.replace('../../pretrained_models', replace_text)
                        if debug == "InspireMusic-Base-24kHz":
                            new_content = new_content.replace('InspireMusic-Base', 'InspireMusic-Base-24kHz')
                        elif debug == "InspireMusic":
                            new_content = new_content.replace('InspireMusic/InspireMusic-Base', 'InspireMusic/InspireMusic')
                        # 将替换后的内容写回文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"已成功替换 {file_path} 中的文本。")
                except Exception as e:
                    print(f"处理 {file_path} 时出现错误: {e}")