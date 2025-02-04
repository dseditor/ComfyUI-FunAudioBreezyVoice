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
import logging
import random

import pyarrow.parquet as pq
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import re

torchaudio.set_audio_backend('soundfile')

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}
CHORUS = {"intro": 0, "chorus": 1, "verse1": 2, "verse2": 3, "verse": 2,
		  "outro": 4}

metadata_pattern = re.compile(r'^\[(ti|ar|al|by|offset):.*\]$')
timestamp_pattern = re.compile(r'^\[\d{2}:\d{2}\.\d{2}\](.*)$')


def parquet_opener(data, mode='train', audio_data={}):
	""" Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
	for sample in data:
		assert 'src' in sample

		url = sample['src']
		try:
			df = pq.read_table(url).to_pandas()
			for i in df.index:
				sample.update(dict(df.loc[i]))
				yield {**sample}
		except Exception as ex:
			logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def clean_lyrics(data, mode="train"):
	for sample in data:
		lyrics = sample["text"]
		cleaned = []
		for line in lyrics.splitlines():
			if metadata_pattern.match(line):
				continue
			timestamp_match = timestamp_pattern.match(line)
			if timestamp_match:
				lyric = timestamp_match.group(1).strip()
				if lyric:
					cleaned.append(lyric)
			else:
				if line.strip():
					cleaned.append(line.strip())
		sample["text"] = '\n'.join(cleaned)
		yield sample


def cut_by_length(data, max_length=8000, num_times=4, mode="train"):
	for sample in data:
		if "semantic_token" in sample:
			sample["semantic_token"] = [
				sample["semantic_token"][0][:max_length]]
		if "acoustic_token" not in sample:
			sample["acoustic_token"] = sample["speech_token"]
		sample["acoustic_token"] = sample["acoustic_token"][
								   :max_length * num_times]

		yield sample


def filter(data,
           max_length=22500,  # 22500 #5min #10240
           max_acoustic_length=45000,
           min_length=10,
           min_acoustic_length=150,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
	""" Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
	if mode == "train":
		for sample in data:
			if "semantic_token" in sample:
				new_sample_frames = sample['semantic_token'][0].shape[0]
			else:
				new_sample_frames = sample['speech_token']

			if "text_token" in sample:
				new_sample_frames += len(sample['text_token'])

			if new_sample_frames > max_length or new_sample_frames < min_length:
				print(f"skipped 1 item length={new_sample_frames}")
				continue

			sample["chorus"] = sample["chorus"].split(",")
			if not isinstance(sample["time_start"], np.ndarray):
				sample["time_start"] = [sample["time_start"]]
				sample["time_end"] = [sample["time_end"]]
			for i, t in enumerate(sample["chorus"]):
				if sample["chorus"][i] == "verse":
					sample["chorus"][i] = "verse1"

			yield sample

	if mode == "train_flow":
		for sample in data:
			if "semantic_token" in sample:
				new_sample_frames = sample['semantic_token'][0].shape[0]
			if "acoustic_token" in sample:
				target_sample_frames = sample['acoustic_token'][0].shape[0]

			if new_sample_frames > max_length or new_sample_frames < min_acoustic_length or new_sample_frames < min_length or target_sample_frames > max_acoustic_length:
				print(
					f"skipped 1 item length={new_sample_frames}, target_length={target_sample_frames}")
				continue

			yield sample

	elif mode == "inference":
		for sample in data:
			yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
	""" Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
	for sample in data:
		assert 'sample_rate' in sample
		assert 'speech' in sample
		sample_rate = sample['sample_rate']
		waveform = sample['speech']
		if sample_rate != resample_rate:
			if sample_rate < min_sample_rate:
				continue
			sample['sample_rate'] = resample_rate
			sample['speech'] = torchaudio.transforms.Resample(
					orig_freq=sample_rate, new_freq=resample_rate)(waveform)
		max_val = sample['speech'].abs().max()
		if max_val > 1:
			sample['speech'] /= max_val
		yield sample


def truncate(data, truncate_length=24576, mode='train'):
	""" Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
	for sample in data:
		waveform = sample['audio']
		if waveform.shape[1] > truncate_length:
			start = random.randint(0, waveform.shape[1] - truncate_length)
			waveform = waveform[:, start: start + truncate_length]
		else:
			waveform = torch.concat([waveform, torch.zeros(1, truncate_length -
														   waveform.shape[1])],
									dim=1)
		sample['audio'] = waveform
		yield sample


def upsample(data, resample_rate=48000, min_sample_rate=16000, mode='train',
			 n_codebook=4):
	""" Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
	for sample in data:
		assert 'semantic_token' in sample
		# TODO: unify data processing key names
		if 'acoustic_token' not in sample:
			continue

		if 'sample_rate' in sample.keys():
			sample_rate = sample['sample_rate']
		else:
			sample_rate = 24000
		token = np.array(sample['semantic_token'][0][:-1])

		# Calculate the repetition factor for resampling
		repetition_factor = int(n_codebook * resample_rate / sample_rate)
		if sample_rate != resample_rate:
			if sample_rate < min_sample_rate:
				continue
			sample['sample_rate'] = resample_rate
			sample['semantic_token'] = np.array(
					[np.repeat(token, repetition_factor)])

		yield sample

def compute_fbank(data,
				  feat_extractor,
				  mode='train'):
	""" Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
	for sample in data:
		assert 'sample_rate' in sample
		assert 'speech' in sample
		assert 'utt' in sample
		assert 'text_token' in sample
		waveform = sample['speech']
		mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
		sample['speech_feat'] = mat
		del sample['speech']
		yield sample


def parse_embedding(data, normalize, mode='train'):
	""" Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """

	for sample in data:
		sample['utt_embedding'] = torch.tensor(sample['utt_embedding'],
											   dtype=torch.float32)
		sample['spk_embedding'] = torch.tensor(sample['spk_embedding'],
											   dtype=torch.float32)
		if normalize:
			sample['utt_embedding'] = F.normalize(sample['utt_embedding'],
												  dim=0)
			sample['spk_embedding'] = F.normalize(sample['spk_embedding'],
												  dim=0)
		yield sample

def tokenize(data, get_tokenizer, allowed_special, mode='train'):
	""" Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
	tokenizer = get_tokenizer()

	for sample in data:
		assert 'text' in sample
		sample['text_token'] = tokenizer.encode(sample['text'],
												allowed_special=allowed_special)
		yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
	""" Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
	buf = []
	for sample in data:
		buf.append(sample)
		if len(buf) >= shuffle_size:
			random.shuffle(buf)
			for x in buf:
				yield x
			buf = []
	# The sample left over
	random.shuffle(buf)
	for x in buf:
		yield x


def sort(data, sort_size=500, mode='train'):
	""" Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

	buf = []
	for sample in data:
		if sample["chorus"] == "verse":
			sample["chorus"] = "verse1"

		if sample["acoustic_token"].shape[0] == 1:
			sample["acoustic_token"] = np.concatenate(
					sample["acoustic_token"][0])
		else:
			sample["acoustic_token"] = np.concatenate(sample["acoustic_token"])

		sample["acoustic_token"] = torch.from_numpy(sample["acoustic_token"])
		buf.append(sample)
		if len(buf) >= sort_size:
			buf.sort(key=lambda x: x['acoustic_token'].size(0))
			for x in buf:
				yield x
			buf = []
	# The sample left over
	buf.sort(key=lambda x: x['acoustic_token'].size(0))
	for x in buf:
		yield x


def static_batch(data, batch_size=32):
	""" Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
	buf = []
	data_empty = True
	for sample in data:
		data_empty = False
		buf.append(sample)
		if len(buf) >= batch_size:
			yield buf
			buf = []
	if data_empty:
		raise ValueError("data is empty")
	if len(buf) > 0:
		yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
	""" Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
	buf = []
	longest_frames = 0
	for sample in data:
		assert 'acoustic_token' in sample
		assert isinstance(sample['acoustic_token'], torch.Tensor)

		if 'semantic_token' in sample:
			new_sample_frames = sample['semantic_token'][0].shape[0]
		else:
			new_sample_frames = sample['semantic_token']

		if "text_token" in sample:
			new_sample_frames += len(sample['text_token'])

		longest_frames = max(longest_frames, new_sample_frames)
		frames_after_padding = longest_frames * (len(buf) + 1)

		if frames_after_padding > max_frames_in_batch:
			if len(buf) > 0:
				yield buf
			buf = [sample]
			longest_frames = new_sample_frames
		else:
			buf.append(sample)
	if len(buf) > 0:
		yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000,
		  mode='train'):
	""" Wrapper for static/dynamic batch
    """
	if mode == 'inference':
		return static_batch(data, 1)
	elif mode == 'processing':
		return static_batch(data, batch_size)
	else:
		if batch_type == 'static':
			return static_batch(data, batch_size)
		elif batch_type == 'dynamic':
			return dynamic_batch(data, max_frames_in_batch)
		else:
			logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, mode='train'):
	""" Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
	if mode == "train":
		for sample in data:
			assert isinstance(sample, list)
			if len(sample) != 0:
				acoustic_feat_len = torch.tensor(
						[x['acoustic_token'].size(0) for x in sample],
						dtype=torch.int32)
				order = torch.argsort(acoustic_feat_len, descending=True)
				utts = [sample[i]['utt'] for i in order]
				acoustic_token = [
					sample[i]['acoustic_token'].clone().to(torch.int32) for i in
					order]
				acoustic_token_len = torch.tensor(
						[i.size(0) for i in acoustic_token], dtype=torch.int32)

				acoustic_token = pad_sequence(acoustic_token,
											  batch_first=True,
											  padding_value=0)

				text = [sample[i]['text'] for i in order]
				text_token = [torch.tensor(sample[i]['text_token']).long() for i
							  in order]
				text_token_len = torch.tensor([i.size(0) for i in text_token],
											  dtype=torch.int32)
				text_token = pad_sequence(text_token, batch_first=True,
										  padding_value=0)
				time_start = torch.tensor(
						[sample[i]['time_start'] for i in order])
				time_end = torch.tensor([sample[i]['time_end'] for i in order])

				if isinstance(sample[0]['chorus'], str):
					chorus = torch.tensor(
							[CHORUS[sample[i]['chorus']] for i in order])
				else:
					chorus = [
						torch.tensor([CHORUS[t] for t in sample[i]['chorus']])
						for i in order]
					chorus = pad_sequence(chorus, batch_first=True,
										  padding_value=-1)

				batch = {
					"utts"              : utts,
					"acoustic_token"    : acoustic_token,
					"acoustic_token_len": acoustic_token_len,
					"time_start"        : time_start,
					"time_end"          : time_end,
					"chorus"            : chorus,
					"text"              : text,
					"text_token"        : text_token,
					"text_token_len"    : text_token_len,
				}

				if "semantic_token" in sample[0]:
					semantic_token = [
						torch.tensor(sample[i]['semantic_token'][0],
									 dtype=torch.int32) for i in order]
					semantic_token_len = torch.tensor(
							[i.size(0) for i in semantic_token],
							dtype=torch.int32)
					semantic_token = pad_sequence(semantic_token,
												  batch_first=True,
												  padding_value=0)
					batch.update({"semantic_token"    : semantic_token,
								  "semantic_token_len": semantic_token_len})

				yield batch
			else:
				logging.info("WARNING: sample is empty []!")

	elif mode == "inference":
		for sample in data:
			assert isinstance(sample, list)
			utts = [sample[i]['utt'] for i in range(len(sample))]
			text = [sample[i]['text'] for i in range(len(sample))]
			text_token = [torch.tensor(sample[i]['text_token']).long() for i in
						  range(len(sample))]
			text_token_len = torch.tensor([i.size(0) for i in text_token],
										  dtype=torch.int32)
			text_token = pad_sequence(text_token, batch_first=True,
									  padding_value=0)
			time_start = torch.tensor(
					[sample[i]['time_start'] for i in range(len(sample))])
			time_end = torch.tensor(
					[sample[i]['time_end'] for i in range(len(sample))])

			if isinstance(sample[0]['chorus'], str):
				chorus = torch.tensor([CHORUS[sample[i]['chorus']] for i in
									   range(len(sample))])
			else:
				chorus = [torch.tensor([CHORUS[t] for t in sample[i]['chorus']])
						  for i in range(len(sample))]
				chorus = pad_sequence(chorus, batch_first=True,
									  padding_value=-1)

			if "acoustic_token" in sample[0]:
				acoustic_token = [
					sample[i]['acoustic_token'].clone().to(torch.int32) for i in
					range(len(sample))]
				acoustic_token_len = torch.tensor(
						[i.size(0) for i in acoustic_token], dtype=torch.int32)
				acoustic_token = pad_sequence(acoustic_token,
											  batch_first=True,
											  padding_value=0)
			else:
				acoustic_token = None
				acoustic_token_len = None

			batch = {
				"utts"              : utts,
				"acoustic_token"    : acoustic_token,
				"acoustic_token_len": acoustic_token_len,
				"time_start"        : time_start,
				"time_end"          : time_end,
				"chorus"            : chorus,
				"text"              : text,
				"text_token"        : text_token,
				"text_token_len"    : text_token_len,
			}

			if "semantic_token" in sample[0]:
				semantic_token = [torch.tensor(sample[i]['semantic_token'][0],
											   dtype=torch.int32) for i in
								  range(len(sample))]
				semantic_token_len = torch.tensor(
						[i.size(0) for i in semantic_token], dtype=torch.int32)
				semantic_token = pad_sequence(semantic_token,
											  batch_first=True,
											  padding_value=0)
				batch.update({"semantic_token"    : semantic_token,
							  "semantic_token_len": semantic_token_len})

			yield batch
