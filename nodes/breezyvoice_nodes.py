'''
Author: SpenserCai
Date: 2024-10-04 12:13:28
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-05 12:23:01
Description: BreezyVoice nodes for ComfyUI
'''
import os
import folder_paths
import numpy as np
import torch
import time
from funaudio_utils.pre import FunAudioLLMTool
from funaudio_utils.download_models import get_speaker_default_path
from funaudio_utils.cosyvoice_plus import CosyVoice1, TextReplacer
from cosyvoice.utils.common import set_all_random_seed

fAudioTool = FunAudioLLMTool()

CATEGORY_NAME = "FunAudioLLM_V2/BreezyVoice"

# 確保BreezyVoice資料夾路徑存在
folder_paths.add_model_folder_path("BreezyVoice", os.path.join(folder_paths.models_dir, "CosyVoice", "BreezyVoice"))

# 全域模型快取，避免重複載入
_model_cache = {}

def get_cached_model(model_dir):
    """
    獲取快取的模型，避免重複載入
    """
    if model_dir not in _model_cache:
        print(f"[BreezyVoice] Loading model into cache: {model_dir}")
        _model_cache[model_dir] = CosyVoice1(model_dir)
        
        # 最佳化模型設定 - 更安全的方式
        model = _model_cache[model_dir]
        
        # 設定為評估模式
        try:
            if hasattr(model, 'model'):
                if hasattr(model.model, 'eval'):
                    model.model.eval()
            elif hasattr(model, 'eval'):
                model.eval()
        except Exception as e:
            print(f"[BreezyVoice] Could not set eval mode: {e}")
        
        # GPU移動 - 更相容的方式
        if torch.cuda.is_available():
            try:
                print(f"[BreezyVoice] Attempting to move model to GPU")
                # 嘗試不同的GPU移動方式
                if hasattr(model, 'to'):
                    model = model.to('cuda')
                elif hasattr(model, 'model') and hasattr(model.model, 'to'):
                    model.model = model.model.to('cuda')
                print(f"[BreezyVoice] Successfully moved model to GPU")
            except Exception as e:
                print(f"[BreezyVoice] Could not move to GPU (will use CPU): {e}")
    else:
        print(f"[BreezyVoice] Using cached model: {model_dir}")
    
    return _model_cache[model_dir]

def return_audio(output, t0, spk_model):
    """
    處理音訊輸出的通用函式 - 最佳化版本
    """
    output_list = []
    
    # 使用更高效的張量操作
    with torch.no_grad():
        for out_dict in output:
            output_numpy = out_dict['tts_speech'].squeeze(0).cpu().numpy() * 32768 
            output_numpy = output_numpy.astype(np.int16)
            output_list.append(torch.from_numpy(output_numpy.astype(np.float32) / 32768).unsqueeze(0))
    
    t1 = time.time()
    inference_time = t1 - t0
    print(f"[BreezyVoice] Inference time: {inference_time:.3f}s")
    
    # 更高效的拼接
    if len(output_list) > 1:
        audio_tensor = torch.cat(output_list, dim=1).unsqueeze(0)
    else:
        audio_tensor = output_list[0].unsqueeze(0)
    
    audio = {"waveform": audio_tensor, "sample_rate": fAudioTool.target_sr}
    
    if spk_model is not None:
        return (audio, spk_model,)
    else:
        return (audio,)

class BreezyVoiceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tts_text": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647
                }),
                "text_frontend": ("BOOLEAN", {
                    "default": True
                }),
                "polyreplace": ("BOOLEAN", {
                    "default": False
                }),
                "optimization_mode": (["quality", "balanced", "speed"], {
                    "default": "balanced",
                    "tooltip": "最佳化模式: quality(品質優先), balanced(平衡), speed(速度優先)"
                }),
                "enable_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "啟用模型快取加速後續推理"
                }),
                "enable_chunking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "啟用chunk處理以支援長文字（超過100字符自動分割）"
                }),
                "max_chars_per_chunk": ("INT", {
                    "default": 80,
                    "min": 40,
                    "max": 200,
                    "step": 10,
                    "tooltip": "每個chunk的最大字符數，建議80字符以內"
                }),
                "crossfade_ms": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "tooltip": "音訊chunk間的交叉淡化時間（毫秒）"
                }),
            },
            "optional": {
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "prompt_wav": ("AUDIO",),
                "speaker_model": ("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "SPK_MODEL",)
    RETURN_NAMES = ("audio", "speaker_model",)
    FUNCTION = "generate"
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def get_optimization_settings(self, mode):
        """
        根據最佳化模式返回設定
        """
        settings = {
            "quality": {
                "use_fp16": False,
                "use_compile": False,
                "fast_inference": False
            },
            "balanced": {
                "use_fp16": True,
                "use_compile": False,
                "fast_inference": True
            },
            "speed": {
                "use_fp16": True,
                "use_compile": True,
                "fast_inference": True
            }
        }
        return settings.get(mode, settings["balanced"])

    def download_breezy_voice_hf(self):
        """
        專門用於從Hugging Face下載BreezyVoice模型
        """
        model_name = "BreezyVoice"
        model_id = "dseditor/BreezyVoice"
        model_dir = os.path.join(folder_paths.models_dir, "CosyVoice", model_name)
        
        if not os.path.exists(model_dir):
            print(f"[BreezyVoice] Downloading model from Hugging Face: {model_id}...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"[BreezyVoice] Model successfully downloaded to {model_dir}")
            except ImportError:
                raise Exception("huggingface_hub is required for BreezyVoice. Please install: pip install huggingface_hub")
            except Exception as e:
                raise Exception(f"Failed to download BreezyVoice model: {e}")
        else:
            print(f"[BreezyVoice] Using existing model at {model_dir}")
        
        return model_dir

    def optimize_inference_settings(self, cosyvoice, use_fp16):
        """
        最佳化推理設定 - 相容實際CosyVoice結構
        """
        actual_fp16 = False
        
        if torch.cuda.is_available():
            # 設定CUDA最佳化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 設定為推理模式 - 更安全的方式
        try:
            if hasattr(cosyvoice, 'model'):
                if hasattr(cosyvoice.model, 'eval'):
                    cosyvoice.model.eval()
            elif hasattr(cosyvoice, 'eval'):
                cosyvoice.eval()
        except Exception as e:
            print(f"[BreezyVoice] Could not set eval mode: {e}")
        
        # FP16最佳化 - 保守方式，主要依賴autocast
        if use_fp16 and torch.cuda.is_available():
            try:
                # 檢查模型結構
                model_attrs = dir(cosyvoice)
                print(f"[BreezyVoice] CosyVoice attributes: {[attr for attr in model_attrs if not attr.startswith('_')][:10]}...")
                
                # 不強制轉換權重，只使用autocast
                print("[BreezyVoice] Will use mixed precision with autocast")
                actual_fp16 = True
                
            except Exception as e:
                print(f"[BreezyVoice] FP16 setup failed: {e}, using FP32")
                actual_fp16 = False
        
        return cosyvoice, actual_fp16
    
    def check_fp16_compatibility(self, model):
        """
        簡化的相容性檢查
        """
        try:
            # 基礎檢查 - 只檢查是否有CUDA支援
            if torch.cuda.is_available():
                print(f"[BreezyVoice] CUDA available, mixed precision supported")
                return True
            else:
                print(f"[BreezyVoice] No CUDA, will use CPU")
                return False
        except Exception as e:
            print(f"[BreezyVoice] Compatibility check failed: {e}")
            return False
    
    def apply_safe_fp16(self, model):
        """
        移除直接的權重轉換，只返回原模型
        """
        print("[BreezyVoice] Skipping weight conversion, using autocast only")
        return model

    def split_text_to_chunks(self, text, max_chars_per_chunk=100):
        """
        智慧分割文字為chunks，避免超過100字符限制
        """
        if len(text) <= max_chars_per_chunk:
            return [text]
        
        chunks = []
        # 優先按句號分割
        sentences = text.split('。')
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 重新加上句號（除了最後一句）
            if i < len(sentences) - 1:
                sentence += '。'
            
            # 檢查加入這句話後是否會超過限制
            if len(current_chunk + sentence) <= max_chars_per_chunk:
                current_chunk += sentence
            else:
                # 如果當前chunk不為空，先保存
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果單句話就超過限制，需要按字符強制分割
                if len(sentence) > max_chars_per_chunk:
                    # 強制按字符分割長句
                    for j in range(0, len(sentence), max_chars_per_chunk):
                        chunk_part = sentence[j:j + max_chars_per_chunk]
                        chunks.append(chunk_part)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # 添加最後一個chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def concatenate_audio_chunks(self, audio_chunks, crossfade_ms=30):
        """
        平滑合併音訊chunks，使用較短的交叉淡化避免聲音變化
        """
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        sample_rate = 24000  # BreezyVoice使用24kHz
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        
        result = audio_chunks[0]
        
        for chunk in audio_chunks[1:]:
            if result.shape[-1] > crossfade_samples and chunk.shape[-1] > crossfade_samples:
                # 創建淡入淡出效果
                fade_out = torch.linspace(1, 0, crossfade_samples, device=result.device)
                fade_in = torch.linspace(0, 1, crossfade_samples, device=chunk.device)
                
                # 應用淡出到result的尾部
                result_fade = result.clone()
                result_fade[..., -crossfade_samples:] *= fade_out
                
                # 應用淡入到chunk的頭部
                chunk_fade = chunk.clone()
                chunk_fade[..., :crossfade_samples] *= fade_in
                
                # 重疊相加
                overlap_part = result_fade[..., -crossfade_samples:] + chunk_fade[..., :crossfade_samples]
                
                # 合併：result前部 + 重疊部分 + chunk後部
                result = torch.cat([
                    result[..., :-crossfade_samples], 
                    overlap_part, 
                    chunk[..., crossfade_samples:]
                ], dim=-1)
            else:
                # 如果太短無法crossfade，直接拼接
                result = torch.cat([result, chunk], dim=-1)
        
        return result

    def chunk_inference_with_reference(self, text_chunks, prompt_text, prompt_speech_16k, 
                                     speed, text_frontend, use_mixed_precision):
        """
        使用重複參考法處理多個文字chunks
        每個chunk都使用原始參考音訊，確保聲音一致性
        """
        all_outputs = []
        total_chunks = len(text_chunks)
        
        print(f"[BreezyVoice] Starting chunked inference for {total_chunks} chunks")
        
        for i, chunk_text in enumerate(text_chunks):
            print(f"[BreezyVoice] Processing chunk {i+1}/{total_chunks}: '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}'")
            
            # 每個chunk都使用相同的參考音訊和prompt文字
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.cosyvoice.inference_zero_shot(
                        chunk_text,
                        prompt_text,  # 保持一致的prompt文字
                        prompt_speech_16k,  # 每次都使用原始參考音訊
                        False, 
                        speed, 
                        text_frontend
                    )
            else:
                output = self.cosyvoice.inference_zero_shot(
                    chunk_text,
                    prompt_text,
                    prompt_speech_16k,
                    False,
                    speed,
                    text_frontend
                )
            
            # 提取生成的音訊 - 處理generator返回值
            if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                # 如果是generator，轉換為list
                output = list(output)
            
            if isinstance(output, (list, tuple)):
                chunk_audio = output[0]['tts_speech']
            else:
                chunk_audio = output['tts_speech']
            
            all_outputs.append(chunk_audio)
            
            # 立即清理GPU快取避免記憶體累積
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[BreezyVoice] Chunk {i+1} completed, audio shape: {chunk_audio.shape}")
        
        return all_outputs

    def preprocess_text(self, tts_text, polyreplace):
        """
        預處理文字
        """
        tts_text = tts_text.strip()
        if polyreplace:
            print("[BreezyVoice] Applying polyphonic word replacement...")
            tts_text = TextReplacer.replace_tts_text(tts_text)
        
        return tts_text

    def generate(self, tts_text, speed, seed, text_frontend, polyreplace, optimization_mode, enable_cache, enable_chunking, max_chars_per_chunk, crossfade_ms, prompt_text=None, prompt_wav=None, speaker_model=None):
        t0 = time.time()
        
        # 驗證輸入
        assert len(tts_text.strip()) > 0, "tts_text不能為空！！！"
        
        # 獲取最佳化設定
        opt_settings = self.get_optimization_settings(optimization_mode)
        print(f"[BreezyVoice] Using {optimization_mode} optimization mode")
        
        # 預處理文字
        tts_text = self.preprocess_text(tts_text, polyreplace)
        
        # 檢查是否需要chunk處理
        need_chunking = enable_chunking and len(tts_text) > max_chars_per_chunk
        
        if need_chunking:
            print(f"[BreezyVoice] Text length ({len(tts_text)} chars) exceeds chunk limit ({max_chars_per_chunk}), enabling chunked processing")
            return self.generate_chunked(tts_text, speed, seed, text_frontend, 
                                       optimization_mode, enable_cache, max_chars_per_chunk,
                                       crossfade_ms, prompt_text, prompt_wav, speaker_model)
        else:
            print(f"[BreezyVoice] Text length ({len(tts_text)} chars) within limit, using standard processing")
            return self.generate_standard(tts_text, speed, seed, text_frontend,
                                        optimization_mode, enable_cache, prompt_text, prompt_wav, speaker_model)

    def generate_standard(self, tts_text, speed, seed, text_frontend, optimization_mode, enable_cache, prompt_text=None, prompt_wav=None, speaker_model=None):
        """
        標準單次推理模式（原有邏輯）
        """
        t0 = time.time()
        
        # 獲取最佳化設定
        opt_settings = self.get_optimization_settings(optimization_mode)
        print(f"[BreezyVoice] Using {optimization_mode} optimization mode")
        
        # 獲取模型目錄
        model_dir = self.download_breezy_voice_hf()
        
        # 根據快取設定獲取模型
        if enable_cache:
            cosyvoice = get_cached_model(model_dir)
        else:
            print(f"[BreezyVoice] Loading model without cache: {model_dir}")
            cosyvoice = CosyVoice1(model_dir)
        
        # 最佳化推理設定
        cosyvoice, use_mixed_precision = self.optimize_inference_settings(cosyvoice, opt_settings["use_fp16"])
        
        # 設定隨機種子
        set_all_random_seed(seed)
        
        # 選擇推理上下文
        if opt_settings["fast_inference"]:
            inference_context = torch.inference_mode()
        else:
            inference_context = torch.no_grad()
        
        with inference_context:
            if speaker_model is None:
                # 使用音訊提示進行零樣本複製
                assert prompt_wav is not None, "當未提供speaker_model時，必須提供prompt_wav！"
                assert len(prompt_text.strip()) > 0, "prompt文字為空，您是否忘記輸入prompt文字？"
                
                # 音訊處理
                speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
                prompt_speech_16k = fAudioTool.postprocess(speech)
                
                print('[BreezyVoice] Zero-shot inference with audio prompt')
                
                # 執行推理 - 使用混合精度
                if use_mixed_precision and torch.cuda.is_available():
                    print("[BreezyVoice] Using mixed precision autocast")
                    with torch.cuda.amp.autocast(enabled=True):
                        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, False, speed, text_frontend)
                        # 處理generator返回值
                        if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                            output = list(output)
                        spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, 24000)
                else:
                    print("[BreezyVoice] Using standard precision")
                    output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, False, speed, text_frontend)
                    # 處理generator返回值
                    if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                        output = list(output)
                    spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, 24000)
                
                # 清理說話人模型
                if 'text' in spk_model:
                    del spk_model['text']
                if 'text_len' in spk_model:
                    del spk_model['text_len']
                
                return return_audio(output, t0, spk_model)
            else:
                # 使用已有的說話人模型
                print('[BreezyVoice] Zero-shot inference with existing speaker model')
                
                if use_mixed_precision and torch.cuda.is_available():
                    print("[BreezyVoice] Using mixed precision autocast")
                    with torch.cuda.amp.autocast(enabled=True):
                        output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model, False, speed, text_frontend)
                        # 處理generator返回值
                        if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                            output = list(output)
                else:
                    print("[BreezyVoice] Using standard precision")
                    output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model, False, speed, text_frontend)
                    # 處理generator返回值
                    if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                        output = list(output)
                
                return return_audio(output, t0, None)

    def generate_chunked(self, tts_text, speed, seed, text_frontend, optimization_mode, enable_cache, max_chars_per_chunk, crossfade_ms, prompt_text=None, prompt_wav=None, speaker_model=None):
        """
        分chunk生成長文字語音（重複參考法）
        """
        t0 = time.time()
        
        # 獲取最佳化設定
        opt_settings = self.get_optimization_settings(optimization_mode)
        print(f"[BreezyVoice] Using {optimization_mode} optimization mode for chunked processing")
        
        # 分割文字為chunks
        text_chunks = self.split_text_to_chunks(tts_text, max_chars_per_chunk)
        print(f"[BreezyVoice] Split text into {len(text_chunks)} chunks: {[len(chunk) for chunk in text_chunks]} chars each")
        
        # 獲取模型目錄
        model_dir = self.download_breezy_voice_hf()
        
        # 根據快取設定獲取模型
        if enable_cache:
            self.cosyvoice = get_cached_model(model_dir)
        else:
            print(f"[BreezyVoice] Loading model without cache: {model_dir}")
            self.cosyvoice = CosyVoice1(model_dir)
        
        # 最佳化推理設定
        self.cosyvoice, use_mixed_precision = self.optimize_inference_settings(self.cosyvoice, opt_settings["use_fp16"])
        
        # 設定隨機種子
        set_all_random_seed(seed)
        
        # 處理音訊和生成chunks
        if speaker_model is None:
            # 使用音訊提示進行零樣本複製
            assert prompt_wav is not None, "當未提供speaker_model時，必須提供prompt_wav！"
            assert len(prompt_text.strip()) > 0, "prompt文字為空，您是否忘記輸入prompt文字？"
            
            # 音訊處理
            speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
            prompt_speech_16k = fAudioTool.postprocess(speech)
            
            print('[BreezyVoice] Chunked zero-shot inference with audio prompt')
            
            # 使用重複參考法處理所有chunks
            audio_chunks = self.chunk_inference_with_reference(
                text_chunks, prompt_text, prompt_speech_16k, 
                speed, text_frontend, use_mixed_precision
            )
            
            # 為第一個chunk生成speaker model
            with torch.no_grad():
                if use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=True):
                        spk_model = self.cosyvoice.frontend.frontend_zero_shot(text_chunks[0], prompt_text, prompt_speech_16k, 24000)
                else:
                    spk_model = self.cosyvoice.frontend.frontend_zero_shot(text_chunks[0], prompt_text, prompt_speech_16k, 24000)
                
                # 清理說話人模型
                if 'text' in spk_model:
                    del spk_model['text']
                if 'text_len' in spk_model:
                    del spk_model['text_len']
            
            speaker_model_output = spk_model
        else:
            # 使用已有的說話人模型
            print('[BreezyVoice] Chunked zero-shot inference with existing speaker model')
            
            audio_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                print(f"[BreezyVoice] Processing chunk {i+1}/{len(text_chunks)} with speaker model")
                
                if use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=True):
                        output = self.cosyvoice.inference_zero_shot_with_spkmodel(chunk_text, speaker_model, False, speed, text_frontend)
                        # 處理generator返回值
                        if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                            output = list(output)
                else:
                    output = self.cosyvoice.inference_zero_shot_with_spkmodel(chunk_text, speaker_model, False, speed, text_frontend)
                    # 處理generator返回值
                    if hasattr(output, '__iter__') and not isinstance(output, (torch.Tensor, dict)):
                        output = list(output)
                
                # 提取音訊
                if isinstance(output, (list, tuple)):
                    audio_chunks.append(output[0]['tts_speech'])
                else:
                    audio_chunks.append(output['tts_speech'])
                
                # 清理記憶體
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            speaker_model_output = None
        
        # 合併音訊chunks
        print(f"[BreezyVoice] Concatenating {len(audio_chunks)} audio chunks with {crossfade_ms}ms crossfade")
        final_audio = self.concatenate_audio_chunks(audio_chunks, crossfade_ms)
        
        # 包裝為返回格式
        output_list = [final_audio]
        
        t1 = time.time()
        inference_time = t1 - t0
        print(f"[BreezyVoice] Chunked inference time: {inference_time:.3f}s for {len(text_chunks)} chunks")
        
        # 更高效的拼接
        if len(output_list) > 1:
            audio_tensor = torch.cat(output_list, dim=1).unsqueeze(0)
        else:
            audio_tensor = output_list[0].unsqueeze(0)
        
        audio = {"waveform": audio_tensor, "sample_rate": fAudioTool.target_sr}
        
        if speaker_model_output is not None:
            return (audio, speaker_model_output,)
        else:
            return (audio,)

class BreezyVoiceCrossLingualNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tts_text": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "prompt_wav": ("AUDIO",),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647
                }),
                "text_frontend": ("BOOLEAN", {
                    "default": True
                }),
                "polyreplace": ("BOOLEAN", {
                    "default": False
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用FP16精度加速推理"
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def download_breezy_voice_hf(self):
        """
        專門用於從Hugging Face下載BreezyVoice模型
        """
        model_name = "BreezyVoice"
        model_id = "dseditor/BreezyVoice"
        model_dir = os.path.join(folder_paths.models_dir, "CosyVoice", model_name)
        
        if not os.path.exists(model_dir):
            print(f"[BreezyVoice] Downloading model from Hugging Face: {model_id}...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"[BreezyVoice] Model successfully downloaded to {model_dir}")
            except ImportError:
                raise Exception("huggingface_hub is required for BreezyVoice. Please install: pip install huggingface_hub")
            except Exception as e:
                raise Exception(f"Failed to download BreezyVoice model: {e}")
        else:
            print(f"[BreezyVoice] Using existing model at {model_dir}")
        
        return model_dir

    def optimize_inference_settings(self, cosyvoice, use_fp16):
        """
        最佳化推理設定
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            if use_fp16 and hasattr(cosyvoice, 'model'):
                print("[BreezyVoice] Using FP16 for faster inference")
                cosyvoice.model = cosyvoice.model.half()
        
        if hasattr(cosyvoice, 'model'):
            cosyvoice.model.eval()
            
        return cosyvoice

    def generate(self, tts_text, prompt_wav, speed, seed, text_frontend, polyreplace, use_fp16):
        t0 = time.time()
        
        # 驗證輸入
        assert len(tts_text.strip()) > 0, "tts_text不能為空！！！"
        assert prompt_wav is not None, "prompt_wav不能為空！！！"
        
        # 預處理文字
        if polyreplace:
            print("[BreezyVoice] Applying polyphonic word replacement...")
            tts_text = TextReplacer.replace_tts_text(tts_text)
        
        # 獲取快取的模型
        model_dir = self.download_breezy_voice_hf()
        cosyvoice = get_cached_model(model_dir)
        
        # 最佳化推理設定
        cosyvoice = self.optimize_inference_settings(cosyvoice, use_fp16)
        
        # 音訊預處理
        speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
        prompt_speech_16k = fAudioTool.postprocess(speech)
        
        print('[BreezyVoice] Cross-lingual inference')
        
        set_all_random_seed(seed)
        
        with torch.inference_mode():
            output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, False, speed, text_frontend)
        
        return return_audio(output, t0, None)