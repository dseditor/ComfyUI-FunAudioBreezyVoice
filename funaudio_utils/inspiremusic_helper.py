from inspiremusic.cli.inference import InspireMusicUnified, set_env_variables

class InspireMusicHelper():
    def music_create(self,task,duration,text,audio_prompt,model_name, model_dir,fast):
        model = InspireMusicUnified(
            model_name = model_name, 
            model_dir = model_dir,
            min_generate_audio_seconds = 1.0,
            max_generate_audio_seconds = duration,
            sample_rate = 24000,
            output_sample_rate = 48000,
            load_jit = True,
            load_onnx = False,
            fast = fast,
            fp16 = True,
            gpu = 0,
            result_dir = '',)
        return model.inference(
            task=task,
            text=text,
            audio_prompt= audio_prompt, # audio prompt file path
            chorus = "verse",
            time_start = 0.0,
            time_end = duration,
            output_fn = "output_audio",
            max_audio_prompt_length = 5.0,
            fade_out_duration = 1.0,
            output_format = "flac",
            fade_out_mode = True,
            trim = False,)