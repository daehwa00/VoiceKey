import os
import json
import random
import typing as tp
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


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
        self.speaker_dict = self._get_speaker_dict()
        self.files = self._get_files()
        print("Successfully loaded {} files".format(len(self.files)))

    def _get_speaker_dict(self) -> tp.Dict[tp.AnyStr, tp.AnyStr]:
        speaker_dict = {}
        for root, dirs, files in os.walk(self.label_dir):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        speaker_name = data["Speaker"]["SpeakerName"]
                        if speaker_name not in speaker_dict:
                            speaker_dict[speaker_name] = os.path.dirname(
                                json_file
                            ).replace(self.label_dir, self.source_dir)
        return speaker_dict

    def _get_files(self) -> tp.List[tp.AnyStr]:
        files = []
        for speaker, dir_path in self.speaker_dict.items():
            files.extend(
                [
                    os.path.relpath(os.path.join(root, f), start=self.source_dir)
                    for root, dirs, files in os.walk(dir_path)
                    for f in files
                    if f.endswith(".pth")
                ]
            )
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, idx: tp.Union[int, torch.Tensor]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load first audio clip
        audio_file1 = os.path.join(self.source_dir, self.files[idx])
        audio1 = torch.load(audio_file1).float() / 1024

        # Randomly select second audio clip
        same_speaker = random.choice([True, False])
        if same_speaker:
            # If same speaker, find another audio clip from the same speaker
            speaker_files = [
                f
                for f in self.files
                if os.path.dirname(f) == os.path.dirname(self.files[idx])
            ]
            audio_file2 = os.path.join(self.source_dir, random.choice(speaker_files))
        else:
            # If not same speaker, select any other audio clip
            audio_file2 = os.path.join(
                self.source_dir,
                random.choice(
                    [
                        f
                        for f in self.files
                        if os.path.dirname(f) != os.path.dirname(self.files[idx])
                    ]
                ),
            )
        audio2 = torch.load(audio_file2).float() / 1024

        return audio1, audio2, torch.tensor(int(same_speaker))


class EcdcDataLoader(DataLoader):
    def __init__(
        self,
        source_dir: tp.AnyStr,
        label_dir: tp.AnyStr,
        batch_size: int = 64,
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
