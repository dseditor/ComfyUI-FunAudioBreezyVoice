from inspiremusic.cli.inference import InspireMusicUnified, set_env_variables

class InspireMusicHelper():
    def music_create(self,task,duration,text,audio_prompt,audio_prompt_rate,model_name, model_dir,fast):
        if task == "continuation" :
            audio_length = audio_prompt.shape[1] / audio_prompt_rate
        else :
            audio_length = 1.0
        model = InspireMusicUnified(
            model_name = model_name,  # 模型名称
            model_dir = model_dir, # 模型路径
            min_generate_audio_seconds = 10.0, # 最小生成音频的长度（秒）
            max_generate_audio_seconds = duration+10.0, # 生成音频的长度（秒）
            sample_rate = 24000, # 输入音频的采样率
            output_sample_rate = 48000, # 输出音频的采样率
            load_jit = True, # 是否使用jit
            load_onnx = False, # 是否使用onnx
            fast = fast, # 是否使用fast模式
            fp16 = True, # 是否使用fp16推理
            gpu = 0, # gpu id
            result_dir = '', # 输出文件路径
            hub = "modelscope", # 模型来源
            )
        return model.inference(
            task=task, # 任务类型
            text=text, # 输入文本
            audio_prompt= audio_prompt,  # 输入音频
            sample_rate = 24000, # 输入音频的采样率
            instruct = None, # 指令
            chorus = "verse", # 副歌标签生成模式，如：随机random、主歌verse、副歌chorus、前奏intro、后奏outro
            time_start = 0.0, # 音频开始时间
            time_end = duration, # 音频长度
            output_fn = "output_audio", # 输出文件名
            max_audio_prompt_length = audio_length, # 音频提示长度
            fade_out_duration = 1.0, # 淡出时间
            output_format = "flac", # 输出格式
            fade_out_mode = True, # 是否淡出
            trim = False, # 是否修剪掉开头和结尾的静音
            )