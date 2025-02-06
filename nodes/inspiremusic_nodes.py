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
    model_list = ['InspireMusic-Base-24kHz','InspireMusic','InspireMusic-1.5B-24kHz','InspireMusic-1.5B','InspireMusic-1.5B-Long']
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model":(s.model_list,{
                    "default":"InspireMusic-Base-24kHz"
                }),
                "text":("STRING", {
                    "default": "",
                    "multiline": True
                }),
            },
            "optional":{
                "prompt_text":("STRING",{
                    "default": "",
                    "multiline": True
                }),
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", )
    FUNCTION="generate"
    
    def generate(self, text,prompt_text=None, prompt_wav=None):
        _, model_dir = download_inspiremusic_1dot5B_long()
        result_dir = folder_paths.output_directory
        generator = InspireMusicHelper()
        music_audio = generator.music_create("continuation",text,None,"InspireMusic-1.5B-Long",model_dir,result_dir,False)
        # if fast sample_rate=24000 else 48000
        audio = {"waveform": music_audio.unsqueeze(0), "sample_rate": 48000}
        return (audio, )