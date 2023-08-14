import os
import json
import random
import typing as tp
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


class EcdcDataset(Dataset):
    def __init__(
        self,
        source_dir: tp.AnyStr,
        label_dir: tp.AnyStr,
        transform: tp.Optional[tp.Callable] = None,
    ):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.transform = transform
        self.speaker_dict, self.all_files = self._get_speaker_and_files()
        self.paired_files = self._create_paired_files()
        print("Successfully created {} paired files".format(len(self.paired_files)))

    def _get_speaker_and_files(
        self,
    ) -> tp.Tuple[tp.Dict[tp.AnyStr, tp.AnyStr], tp.List[tp.AnyStr]]:
        speaker_dict = {}
        files = []

        # os.walk returns a tuple of (dirpath, dirnames, filenames)
        for root, dirs, filenames in os.walk(self.label_dir):
            for file in filenames:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        speaker_name = data["Speaker"]["SpeakerName"]
                        if speaker_name not in speaker_dict:
                            speaker_dict[speaker_name] = os.path.dirname(
                                json_file
                            ).replace(self.label_dir, self.source_dir)

        for root, dirs, filenames in os.walk(self.source_dir):
            for file in filenames:
                if file.endswith(".pth"):
                    rel_path = os.path.relpath(
                        os.path.join(root, file), start=self.source_dir
                    )
                    files.append(rel_path)

        return speaker_dict, files

    def _create_paired_files(self) -> tp.List[tp.Tuple[tp.AnyStr, tp.AnyStr, bool]]:
        paired_files = []
        # Use all available CPUs
        num_processes = cpu_count()
        with Pool(processes=num_processes) as pool:
            paired_files = list(
                tqdm(
                    pool.imap(
                        partial(_pair_files, all_files=self.all_files),
                        range(len(self.all_files)),
                        chunksize=100,
                    ),
                    total=len(self.all_files),
                    desc="Creating paired files",
                )
            )

        return paired_files

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(
        self, idx: tp.Union[int, torch.Tensor]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file1, audio_file2, same_speaker = self.paired_files[idx]

        audio1 = torch.load(os.path.join(self.source_dir, audio_file1)).float() / 1024
        audio2 = torch.load(os.path.join(self.source_dir, audio_file2)).float() / 1024

        return audio1, audio2, torch.tensor(int(same_speaker))


class EcdcDataLoader(DataLoader):
    def __init__(
        self,
        source_dir: tp.AnyStr,
        label_dir: tp.AnyStr,
        batch_size: int = 1024,
        shuffle: bool = True,
        num_workers: int = 24,
        pin_memory: bool = True,
    ):
        self.dataset = EcdcDataset(source_dir, label_dir)
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self) -> tp.Iterator:
        return super().__iter__()

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(
        self, idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__getitem__(idx)


def _pair_files(idx, all_files):
    file = all_files[idx]
    same_speaker = random.choice([True, False])
    if same_speaker:
        speaker_files = [
            f
            for f in all_files
            if os.path.dirname(f) == os.path.dirname(all_files[idx])
        ]
        paired_file = random.choice(speaker_files)
    else:
        paired_file = random.choice(
            [
                f
                for f in all_files
                if os.path.dirname(f) != os.path.dirname(all_files[idx])
            ]
        )
    return (file, paired_file, same_speaker)


class PreprocessedEcdcDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
        print("Successfully loaded {} preprocessed files".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PreprocessedEcdcDataLoader(DataLoader):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 24,
        pin_memory: bool = True,
    ):
        self.dataset = PreprocessedEcdcDataset(data_path)
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
