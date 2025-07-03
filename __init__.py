'''
Author: SpenserCai
Date: 2024-10-04 12:14:22
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 22:29:33
Description: file content
'''
from .nodes.cosyvoice_nodes import *
from .nodes.sensevoice_nodes import *
from .nodes.inspiremusic_nodes import *
from .nodes.breezyvoice_nodes import *

NODE_CONFIG = {
    "CosyVoice2ZeroShotNode": {
        "class": CosyVoice2ZeroShotNode,
        "name": "CosyVoice2 音色克隆"
    },
    "CosyVoice2CrossLingualNode": {
        "class": CosyVoice2CrossLingualNode,
        "name": "CosyVoice2 跨语言克隆"
    },
    "CosyVoice2InstructNode": {
        "class": CosyVoice2InstructNode,
        "name": "CosyVoice2 自然语言控制"
    },
    "CosyVoiceZeroShotNode": {
        "class": CosyVoiceZeroShotNode,
        "name": "CosyVoice 音色克隆"
    },
    "CosyVoiceSFTNode": {
        "class": CosyVoiceSFTNode,
        "name": "CosyVoice 预训练音色"
    },
    "CosyVoiceCrossLingualNode": {
        "class": CosyVoiceCrossLingualNode,
        "name": "CosyVoice 跨语言克隆"
    },
    "CosyVoiceInstructNode": {
        "class": CosyVoiceInstructNode,
        "name": "CosyVoice 自然语言控制"
    },
    "CosyVoiceSaveSpeakerModelNode": {
        "class": CosyVoiceSaveSpeakerModelNode,
        "name": "CosyVoice 保存说话人模型"
    },
    "CosyVoiceLoadSpeakerModelNode": {
        "class": CosyVoiceLoadSpeakerModelNode,
        "name": "CosyVoice 加载说话人模型"
    },
    "CosyVoiceLoadSpeakerModelFromUrlNode": {
        "class": CosyVoiceLoadSpeakerModelFromUrlNode,
        "name": "CosyVoice 从URL加载说话人模型"
    },
    "BreezyVoiceNode": {
        "class": BreezyVoiceNode,
        "name": "BreezyVoice 音色複製"
    },
    "BreezyVoiceCrossLingualNode": {
        "class": BreezyVoiceCrossLingualNode,
        "name": "BreezyVoice 跨語言音色複製"
    },
    "SenseVoiceNode": {
        "class": SenseVoiceNode,
        "name": "SenseVoice 语音识别"
    },
    
    "TextToMusic": {
        "class": TextToMusic,
        "name": "文本生成音乐"
    },
    
    "MusicContinuation": {
        "class": MusicContinuation,
        "name": "音乐延续"
    },
}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]