import torch
from omegaconf import DictConfig
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset

from data.encoders.encoder import MidiEncoder


class MidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        dataset_cfg: DictConfig,
        encoder: MidiEncoder,
    ):
        super().__init__()

        # Dataset with tokenized MIDI data
        self.dataset = dataset
        self.encoder = encoder
        self.dataset_cfg = dataset_cfg

    def __len__(self) -> int:
        return len(self.dataset)

    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]

        # Encoder defines the task
        source_tokens_ids, target_tokens_ids = self.encoder.encode(record)

        out = {
            "source_token_ids": torch.tensor(source_tokens_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_tokens_ids, dtype=torch.int64),
        }
        return out
