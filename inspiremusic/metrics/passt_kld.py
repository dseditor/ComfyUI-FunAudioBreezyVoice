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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import contextlib
from functools import partial
from tqdm import tqdm
import pickle
import numpy as np
import librosa
from hear21passt.base import get_basic_model
import pyloudnorm as pyln

import torch
import torch.nn.functional as F


SAMPLING_RATE = 32000


class _patch_passt_stft:
    """    
    From version 1.8.0, return_complex must always be given explicitly 
    for real inputs and return_complex=False has been deprecated.

    Decorator to patch torch.stft in PaSST that uses an old stft version.

    Adapted from: https://github.com/facebookresearch/audiocraft/blob/a2b96756956846e194c9255d0cdadc2b47c93f1b/audiocraft/metrics/kld.py
    """
    def __init__(self):
        self.old_stft = torch.stft

    def __enter__(self):
        # return_complex is a mandatory parameter in latest torch versions.
        # torch is throwing RuntimeErrors when not set.
        # see: https://pytorch.org/docs/1.7.1/generated/torch.stft.html?highlight=stft#torch.stft
        # see: https://github.com/kkoutini/passt_hear21/commit/dce83183674e559162b49924d666c0a916dc967a
        torch.stft = partial(torch.stft, return_complex=False)

    def __exit__(self, *exc):
        torch.stft = self.old_stft


def return_probabilities(model, audio_path, window_size=10, overlap=5, collect='mean'):
    """
    Given an audio and the PaSST model, return the probabilities of each AudioSet class.

    Audio is converted to mono at 32kHz.

    PaSST model is trained with 10 sec inputs. We refer to this parameter as the window_size.
    We set it to 10 sec for consistency with PaSST training.

    For longer audios, we split audio into overlapping analysis windows of window_size and overlap of 10 and 5 seconds.
    PaSST supports 10, 20 or 30 sec inputs. Not longer inputs: https://github.com/kkoutini/PaSST/issues/19 

    Note that AudioSet taggers normally use sigmoid output layers. Yet, to compute the
    KL we work with normalized probabilities by running a softmax over logits as in MusicGen:
    https://github.com/facebookresearch/audiocraft/blob/a2b96756956846e194c9255d0cdadc2b47c93f1b/audiocraft/metrics/kld.py

    This implementation assumes run will be on GPU.

    Params:
    -- model: PaSST model on a GPU.
    -- audio_path: path to the audio to be loaded with librosa.
    -- window_size (default=10 sec): analysis window (and receptive field) of PaSST.
    -- overlap (default=5 sec): overlap of the running analysis window for inputs longar than window_size (10 sec).
    -- collect (default='mean'): for longer inputs, aggregate/collect via 'mean' or 'max' pooling along logits vector.
    Returns:
    --  527 probabilities (after softmax, no logarithm).
    """
    # load the audio using librosa
    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    audio = pyln.normalize.peak(audio, -1.0)

    # calculate the step size for the analysis windows with the specified overlap
    step_size = int((window_size - overlap) * SAMPLING_RATE)

    # iterate over the audio, creating analysis windows
    probabilities = []
    for i in range(0, max(step_size, len(audio) - step_size), step_size):
        # extract the current analysis window
        window = audio[i:i + int(window_size * SAMPLING_RATE)]

        # pad the window with zeros if it's shorter than the desired window size
        if len(window) < int(window_size * SAMPLING_RATE):
            # discard window if it's too small (avoid mostly zeros predicted as silence), as in MusicGen:
            # https://github.com/facebookresearch/audiocraft/blob/a2b96756956846e194c9255d0cdadc2b47c93f1b/audiocraft/metrics/kld.py
            if len(window) > int(window_size * SAMPLING_RATE * 0.15):
                tmp = np.zeros(int(window_size * SAMPLING_RATE))
                tmp[:len(window)] = window
                window = tmp

        # convert to a PyTorch tensor and move to GPU
        audio_wave = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).cuda()

        # get the probabilities for this analysis window
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            with torch.no_grad(), _patch_passt_stft():
                logits = model(audio_wave)
                probabilities.append(torch.squeeze(logits))

    probabilities = torch.stack(probabilities)
    if collect == 'mean':
        probabilities = torch.mean(probabilities, dim=0)
    elif collect == 'max':
        probabilities, _ = torch.max(probabilities, dim=0)

    return F.softmax(probabilities, dim=0).squeeze().cpu()


def passt_kld(ids, eval_path, eval_files_extension='.wav', ref_path=None, ref_files_extension='.wav', load_ref_probabilities=None, no_ids=[], collect='mean'):
    """
    Compute KL-divergence between the label probabilities of the generated audio with respect to the original audio.
    Both generated audio (in eval_path) and original audio (in ref_path) are represented by the same prompt/description.
    Audios are identified by an id, that is the name of the file in both directories and links the audio with the prompt/description.
    segmenting the audio

    For inputs longer that the 10 sec PaSST was trained on, we aggregate/collect via 'mean' (default) or 'max' pooling along the logits vector.
    We split the inpot into overlapping analysis windows. Subsequently, we aggregate/collect (accross windows) the generated logits and then apply a softmax. 

    This evaluation script assumes that ids are in both ref_path and eval_path.

    We label probabilities via the PaSST model: https://github.com/kkoutini/PaSST

    GPU-based computation.
    
    Extracting the probabilities is timeconsuming. After being computed once, we store them.
    We store pre-computed reference probabilities in load/ 
    To load those and save computation, just set the path in load_ref_probabilities.
    If load_ref_probabilities is set, ref_path is not required.

    Params:
    -- ids: list of ids present in both eval_path and ref_path. 
    -- eval_path: path where the generated audio files to evaluate are available.
    -- eval_files_extenstion: files extension (default .wav) in eval_path.
    -- ref_path: path where the reference audio files are available. (instead of load_ref_probabilities)
    -- ref_files_extenstion: files extension (default .wav) in ref_path.
    -- load_ref_probabilities: path to the reference probabilities. (inestead of ref_path)
    -- no_ids: it is possible that some reference audio is corrupted or not present. Ignore some this list of ids.
    -- collect (default='mean'): for longer inputs, aggregate/collect via 'mean' or 'max' pooling along the logits vector.
    Returns:
    -- KL divergence
    """
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f): # capturing all useless outputs from passt
        # load model
        model = get_basic_model(mode="logits")
        model.eval()
        model = model.cuda()

    if not os.path.isdir(eval_path):
        if not os.path.isfile(eval_path): 
            raise ValueError('eval_path does not exist')

    if load_ref_probabilities:
        if not os.path.exists(load_ref_probabilities):
            raise ValueError('load_ref_probabilities does not exist')     
        print('[LOADING REFERENCE PROBABILITIES] ', load_ref_probabilities)
        with open(load_ref_probabilities, 'rb') as fp:
            ref_p = pickle.load(fp)

    else:
        if ref_path:
            if not os.path.isdir(ref_path):
                if os.path.isfile(ref_path):
                    id2utt = {}
                    with open(ref_path, "r") as f:
                        for line in f:
                            sec = line.strip().split(" ")
                            id2utt[sec[0]] = sec[1]
                    f.close()
                else:
                    raise ValueError("ref_path does not exist")        
            print('[EXTRACTING REFERENCE PROBABILITIES] ', ref_path)
            ref_p = {}
            for id in tqdm(ids):
                if id not in no_ids:
                    try:
                        if os.path.isfile(ref_path):
                            if id in id2utt.keys():
                                audio_path = id2utt[id]
                            else:
                                raise ValueError(f"id: {id} not in {ref_path}!")  
                        else:
                            audio_path = os.path.join(ref_path, str(id)+ref_files_extension)
                        if os.path.isfile(audio_path):
                            ref_p[id] = return_probabilities(model, audio_path, collect=collect)
                    except Exception as e:
                        print(f"An unexpected error occurred with {id}: {e}\nIf you failed to download it you can add it to no_ids list.")

            # store reference probabilities to load later on
            if not os.path.exists('load/passt_kld/'):
                os.makedirs('load/passt_kld/')
            save_ref_probabilities_path = 'load/passt_kld/'+ref_path.replace('/', '_')+'_collect'+str(collect)+'__reference_probabilities.pkl'
            with open(save_ref_probabilities_path, 'wb') as fp:
                pickle.dump(ref_p, fp)        
            print('[REFERENCE EMBEDDINGS][SAVED] ', save_ref_probabilities_path)

        else:
            raise ValueError('Must specify ref_path or load_ref_probabilities')

    print('[EVALUATING GENERATIONS] ', eval_path)
            
    passt_kl = 0
    count = 0
    for id in tqdm(ids):
        if id not in no_ids:
            try:
                audio_path = os.path.join(eval_path, str(id)+eval_files_extension)
                if os.path.isfile(audio_path):
                    eval_p = return_probabilities(model, audio_path, collect=collect)
                    # note: F.kl_div(x, y) is KL(y||x)
                    # see: https://github.com/pytorch/pytorch/issues/7337
                    # see: https://discuss.pytorch.org/t/kl-divergence-different-results-from-tf/56903/2
                    passt_kl += F.kl_div((ref_p[id] + 1e-6).log(), eval_p, reduction='sum', log_target=False)
                    count += 1
            except Exception as e:
                print(f"An unexpected error occurred with {id}: {e}\nIf you failed to download it you can add it to no_ids list.")
    return passt_kl / count if count > 0 else 0
