import hashlib

import torchaudio
from funaudio_utils.download_models import download_inspiremusic_base_24k,download_inspiremusic_base_48k,download_inspiremusic_1dot5B_24k,download_inspiremusic_1dot5B_48k,download_inspiremusic_1dot5B_long
import folder_paths
import os
from funaudio_utils.inspiremusic_helper import InspireMusicHelper

CATEGORY_NAME = "FunAudioLLM_V2/InspireMusic"
folder_paths.add_model_folder_path("InspireMusic", os.path.join(folder_paths.models_dir, "InspireMusic"))

# 文本生成音乐
class TextToMusic:
    model_list = ['InspireMusic-Base-24kHz','InspireMusic-Base','InspireMusic-1.5B-24kHz','InspireMusic-1.5B','InspireMusic-1.5B-Long']
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model":(s.model_list,{
                    "default":"InspireMusic-Base-24kHz"
                }),
                "auto_download":("BOOLEAN",{
                    "default": False
                }),
                "fast":("BOOLEAN",{
                    "default": True
                }),
                "duration":("FLOAT",{
                    "default": 1.0
                }),
                "text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
            },
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", )
    FUNCTION="generate"
    
    def generate(self, model ,fast, duration, text,auto_download):
        if duration < 1.0:
            duration = 1.0
        model_downloader = {
            "InspireMusic-Base-24kHz":download_inspiremusic_base_24k,
            "InspireMusic-Base":download_inspiremusic_base_48k,
            "InspireMusic-1.5B-24kHz":download_inspiremusic_1dot5B_24k,
            "InspireMusic-1.5B":download_inspiremusic_1dot5B_48k,
            "InspireMusic-1.5B-Long":download_inspiremusic_1dot5B_long,
        }
        _, model_dir = model_downloader[model](auto_download)
        generator = InspireMusicHelper()
        music_audio = generator.music_create("text-to-music",duration,text,None,model,model_dir,fast)
        if fast :
            sample_rate=24000 
        else :
            sample_rate=48000
        audio = {"waveform": music_audio.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

# 音乐延续
class MusicContinuation:
    model_list = ['InspireMusic-Base-24kHz','InspireMusic-Base','InspireMusic-1.5B-24kHz','InspireMusic-1.5B','InspireMusic-1.5B-Long']
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model":(s.model_list,{
                    "default":"InspireMusic-Base-24kHz"
                }),
                "auto_download":("BOOLEAN",{
                    "default": False
                }),
                "fast":("BOOLEAN",{
                    "default": True
                }),
                "duration":("FLOAT",{
                    "default": 1.0
                }),
                "prompt_wav": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
            },
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", )
    FUNCTION="generate"

    def generate(self, model ,fast, duration,audio_prompt, text,auto_download):
        if duration < 1.0:
            duration = 1.0
        model_downloader = {
            "InspireMusic-Base-24kHz":download_inspiremusic_base_24k,
            "InspireMusic-Base":download_inspiremusic_base_48k,
            "InspireMusic-1.5B-24kHz":download_inspiremusic_1dot5B_24k,
            "InspireMusic-1.5B":download_inspiremusic_1dot5B_48k,
            "InspireMusic-1.5B-Long":download_inspiremusic_1dot5B_long,
        }
        _, model_dir = model_downloader[model](auto_download)
        generator = InspireMusicHelper()
        music_audio = generator.music_create("continuation",duration,text,audio_prompt,model,model_dir,fast)
        if fast :
            sample_rate=24000 
        else :
            sample_rate=48000
        audio = {"waveform": music_audio.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )
    
# 加载音乐
# class LoadAudioPath:
#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = folder_paths.get_input_directory()
#         files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
#         return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

#     CATEGORY = CATEGORY_NAME

#     RETURN_TYPES = ("STRING", )
#     FUNCTION = "load"

#     def load(self, audio):
#         audio_path = folder_paths.get_annotated_filepath(audio)
#         return (audio_path, )
#     @classmethod
#     def IS_CHANGED(s, audio):
#         image_path = folder_paths.get_annotated_filepath(audio)
#         m = hashlib.sha256()
#         with open(image_path, 'rb') as f:
#             m.update(f.read())
#         return m.digest().hex()

#     @classmethod
#     def VALIDATE_INPUTS(s, audio):
#         if not folder_paths.exists_annotated_filepath(audio):
#             return "Invalid audio file: {}".format(audio)
#         return True

class AudioPath:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

    CATEGORY = CATEGORY_NAME

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio_path, )

    @classmethod
    def IS_CHANGED(s, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True
