# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
import re
import sys
import inspect
import random
import typing as tp
from functools import partial

import omegaconf
import torch
import torchaudio
import numpy as np

from typing_extensions import Literal
from typing import (
    Any,
    Union,
    Iterable,
    List,
    Dict,
    Optional,
    Tuple,
)

from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

_BoolLike_co = Union[bool, np.bool_]
_IntLike_co = Union[_BoolLike_co, int, "np.integer[Any]"]
_FloatLike_co = Union[_IntLike_co, float, "np.floating[Any]"]

def process_audio(file_path, target_sample_rate=24000):
    audio, sample_rate = torchaudio.load(file_path)
    # Check if the audio needs to be resampled
    if sample_rate != target_sample_rate:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(audio)
    # Convert stereo to mono (if necessary)
    audio = audio.mean(dim=0, keepdim=True) if audio.size(0) == 2 else audio
    return audio, target_sample_rate

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    # global mel_basis, hann_window  # pylint: disable=global-statement,global-variable-not-assigned
    mel_basis = {}
    hann_window = {}  
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def fade_out(audio: torch.Tensor, sample_rate: int,
             fade_duration: float) -> torch.Tensor:
    """
    Apply a linear fade-out effect to the given audio waveform.

    Parameters:
    audio (torch.Tensor): The audio waveform tensor.
    sample_rate (int): Sample rate of the audio.
    fade_duration (float): Duration of the fade-out effect in seconds.

    Returns:
    torch.Tensor: The audio with the fade-out effect applied.
    """
    fade_samples = int(fade_duration * sample_rate)

    if fade_samples > audio.shape[1]:
        fade_samples = audio.shape[
            1]  # use the whole length of audio if necessary

    fade_out_envelope = torch.linspace(1.0, 0.0, fade_samples,
                                       dtype=audio.dtype, device=audio.device)

    fade_section = audio[:, -fade_samples:].clone()

    fade_section *= fade_out_envelope

    faded_audio = audio.clone()
    faded_audio[:, -fade_samples:] = fade_section

    return faded_audio

def split_wav_into_chunks(num_samples, wav, max_chunk_size, minimum_chunk_size=720):
    num_chunks = (num_samples + max_chunk_size - 1) // max_chunk_size  # Ceiling division
    wav_chunks = []
    for i in range(num_chunks):
        start_idx = i * max_chunk_size
        end_idx = min(start_idx + max_chunk_size, num_samples)
        if (end_idx - start_idx) >= minimum_chunk_size:
            if len(wav.shape) == 2:
                chunk = wav[:,start_idx:end_idx]
            else:
                chunk = wav[start_idx:end_idx]
            wav_chunks.append(chunk)
        else:
            print(f"{num_samples}:{num_chunks}, chunk size={(end_idx - start_idx)} is lower then minimum_chunk_size!")
    return wav_chunks

def tiny(x: Union[float, np.ndarray]) -> _FloatLike_co:
    """Compute the tiny-value corresponding to an input's data type.
    """
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny

def detect_silence(audio, sample_rate, threshold=0.05, min_silence_duration=1):
    """
    Detects the first occurrence of silence in the audio.

    Parameters:
        audio (Tensor): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        threshold (float): The threshold below which the signal is considered silent.
        min_silence_duration (float): The minimum duration of silence in seconds.

    Returns:
        int: The timestamp (in samples) where the silence starts.
    """
    # Convert the audio to a numpy array for easier manipulation
    audio_np = audio.numpy().flatten()
    # Calculate the energy of the signal
    energy = np.abs(audio_np)
    # Find the indices where the energy is below the threshold
    silent_indices = np.where(energy < threshold)[0]
    # Find the start and end of contiguous silent regions
    silent_regions = np.split(silent_indices, np.where(np.diff(silent_indices) != 1)[0] + 1)
    # Filter out regions that are too short
    min_silence_samples = int(min_silence_duration * sample_rate)
    for region in silent_regions:
        if len(region) >= min_silence_samples:
            return region[0]
    
    # If no silence is found, return the length of the audio
    return len(audio_np)

def trim_audio(waveform, sample_rate=24000, threshold=0.05, min_silence_duration=1, minimum_silence_start_sample=24000):
    """
    Trims the audio from the beginning to the first occurrence of silence.

    Parameters:
        waveform (Tensor): The waveform data to the input audio file.
        sample_rate (int): Sample rate of the input audio file.
        threshold (float): The threshold below which the signal is considered silent.
        min_silence_duration (float): The minimum duration of silence in seconds.
    """
    # Detect the first occurrence of silence
    silence_start_sample = detect_silence(waveform, sample_rate, threshold, min_silence_duration)
    if silence_start_sample > minimum_silence_start_sample :
        trimmed_waveform = waveform[:silence_start_sample]
    else:
        trimmed_waveform = waveform[:minimum_silence_start_sample]
    if isinstance(trimmed_waveform, torch.Tensor):
        return trimmed_waveform
    else:
        return trimmed_waveform.unsqueeze()

def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output

def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[_FloatLike_co] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    """Normalize an array along a chosen axis.
    """
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(f"threshold={threshold} must be strictly positive")

    if fill not in [None, False, True]:
        raise ParameterError(f"fill={fill} must be None or boolean")

    if not np.isfinite(S).all():
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    S = S.numpy()
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S

    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    else:
        raise ParameterError(f"Unsupported norm: {repr(norm)}")

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm

def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to float 32 bits PCM format.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float32 PCM format
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float16 PCM format
    """
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav


def compress(wav: torch.Tensor, sr: int,
             target_format: tp.Literal["mp3", "ogg", "flac"] = "mp3",
             bitrate: str = "128k") -> tp.Tuple[torch.Tensor, int]:
    """Convert audio wave form to a specified lossy format: mp3, ogg, flac

    Args:
        wav (torch.Tensor): Input wav tensor.
        sr (int): Sampling rate.
        target_format (str): Compression format (e.g., 'mp3').
        bitrate (str): Bitrate for compression.

    Returns:
        Tuple of compressed WAV tensor and sampling rate.
    """

    # Extract the bit rate from string (e.g., '128k')
    match = re.search(r"\d+(\.\d+)?", str(bitrate))
    parsed_bitrate = float(match.group()) if match else None
    assert parsed_bitrate, f"Invalid bitrate specified (got {parsed_bitrate})"
    try:
        # Create a virtual file instead of saving to disk
        buffer = io.BytesIO()

        torchaudio.save(
            buffer, wav, sr, format=target_format, bits_per_sample=parsed_bitrate,
        )
        # Move to the beginning of the file
        buffer.seek(0)
        compressed_wav, sr = torchaudio.load(buffer)
        return compressed_wav, sr

    except RuntimeError:
        logger.warning(
            f"compression failed skipping compression: {format} {parsed_bitrate}"
        )
        return wav, sr


def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    """Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    # Convert to MP3 format with specified bitrate
    wav_tensor_flat, _ = compress(wav_tensor_flat, sr, bitrate=bitrate)

    # Reshape back to original batch format and trim or pad if necessary
    wav_tensor = wav_tensor_flat.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]
    if compressed_length > original_length:
        wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
    elif compressed_length < original_length:
        padding = torch.zeros(
            batch_size, channels, original_length - compressed_length, device=device
        )
        wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    # Move tensor back to the original device
    return wav_tensor.to(device)


def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: tp.Optional[int] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    import tempfile
    import subprocess

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "128"
    )  # Default to 128 if parsing fails

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr, backend="ffmpeg")

        # Prepare FFmpeg command for AAC conversion
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        try:
            # Run FFmpeg and suppress output
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load the AAC audio back into a tensor
            aac_tensor, _ = torchaudio.load(output_path, backend="ffmpeg")
        except Exception as exc:
            raise RuntimeError(
                "Failed to run command " ".join(command)} "
                "(Often this means ffmpeg is not installed or the encoder is not supported, "
                "make sure you installed an older version ffmpeg<5)"
            ) from exc

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        aac_tensor = aac_tensor[:, :original_length_flat]

    # Pad the shortedn frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)