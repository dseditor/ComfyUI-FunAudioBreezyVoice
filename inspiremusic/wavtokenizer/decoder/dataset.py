from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import soundfile
# import librosa
import random

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.stack(batch, dim=0)

class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True, collate_fn=collate_fn
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        # y, sr = torchaudio.load(audio_path)
        # print(audio_path,"111")
        try:
            y1, sr = soundfile.read(audio_path)
            # y1, sr = librosa.load(audio_path,sr=None)
            y = torch.tensor(y1).float().unsqueeze(0)
            # if y.size(0) > 1:
            #     # mix to mono
            #     y = y.mean(dim=0, keepdim=True)
            if y.ndim > 2:
                # mix to mono
                # print("有问题哈,数据处理部分")
                # y = y.mean(dim=-1, keepdim=False)
                random_channel = random.randint(0, y.size(-1) - 1)
                y = y[:, :, random_channel] 

            gain = np.random.uniform(-1, -6) if self.train else -3
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            if y.size(-1) < self.num_samples:
                pad_length = self.num_samples - y.size(-1)
                padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            elif self.train:
                start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                y = y[:, start : start + self.num_samples]
            else:
                # During validation, take always the first segment for determinism
                y = y[:, : self.num_samples]

            return y[0]
        except Exception as e:
            print(f"Error processing file {audio_path} at index {index}: {e}")
            # 这里可以继续选择抛出异常，或者返回一个 None 表示无效数据
            return None

    # def __getitem__(self, index: int) -> torch.Tensor:
    #     audio_path = self.filelist[index]
    #     try:
    #         y, sr = torchaudio.load(audio_path)
    #         if y.size(0) > 1:
    #             # 随机选择一个通道
    #             random_channel = random.randint(0, y.size(0) - 1)
    #             y = y[random_channel, :].unsqueeze(0)  # 保持返回值为 (1, T) 的形式
    #         # gain = np.random.uniform(-1, -6) if self.train else -3
    #         # y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
    #         if sr != self.sampling_rate:
    #             y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
    #         if y.size(-1) < self.num_samples:
    #             pad_length = self.num_samples - y.size(-1)
    #             padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
    #             y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
    #         elif self.train:
    #             start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
    #             y = y[:, start: start + self.num_samples]
    #         else:
    #             # During validation, take always the first segment for determinism
    #             y = y[:, :self.num_samples]
    #         return y[0]
    #     except Exception as e:
    #         print(f"Error processing file {audio_path} at index {index}: {e}")
    #         # 这里可以继续选择抛出异常，或者返回一个 None 表示无效数据
    #         return None