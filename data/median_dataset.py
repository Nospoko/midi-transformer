from typing import Literal
from functools import partial
from multiprocessing import Manager

import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset

from data.dataset import MidiDataset
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


class MedianDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: AwesomeTokenizer | ExponentialTokenizer,
        sequence_length: int,
        notes_per_record: int,
        loss_masking: Literal["finetuning", "pretraining"] = "pretraining",
    ):
        super().__init__(dataset=dataset, tokenizer=tokenizer, loss_masking=loss_masking)
        self.sequence_length = sequence_length
        self.notes_per_record = notes_per_record
        self.length = 0
        self.record_lengths = []
        self._build_record_lengths()

    def _build_record_lengths(self):
        def get_length(record, notes_per_record, shared_list):
            length = len(record["notes"]["pitch"]) - notes_per_record + 1
            shared_list.append(length)

        with Manager() as manager:
            shared_list = manager.list()
            get_length_partial = partial(get_length, notes_per_record=self.notes_per_record, shared_list=shared_list)

            self.dataset.map(get_length_partial, num_proc=32, desc="Building record lengths")
            self.record_lengths = list(shared_list)

        self.length = sum(self.record_lengths)

    def __len__(self):
        return self.length

    def _index_to_record_and_start(self, idx):
        for record_id, length in enumerate(self.record_lengths):
            if idx < length:
                return record_id, idx
            idx -= length
        raise IndexError("Index out of range")

    def __getitem__(self, idx: int) -> dict:
        record_id, start_point = self._index_to_record_and_start(idx)

        record = self.dataset[record_id]

        notes = pd.DataFrame(record["notes"])
        notes = notes.iloc[start_point : start_point + self.notes_per_record]

        offset = notes.start.min()
        notes.start = notes.start - offset
        notes.end = notes.end - offset

        median = notes.pitch.median()
        source_notes = notes[notes.pitch < median]
        target_notes = notes[notes.pitch >= median]

        source_prefix = "<LOW_FROM_MEDIAN>"
        target_prefix = "<HIGH_FROM_MEDIAN>"

        prompt_token_ids = self.tokenizer.encode(
            notes=source_notes,
            prefix_tokens=[source_prefix],
        )
        target_token_ids = self.tokenizer.encode(
            notes=target_notes,
            prefix_tokens=[target_prefix],
        )
        encoding = prompt_token_ids + target_token_ids
        # Concatenating tokens so had to move padding here again
        # I think this is good place for paddig btw, because we do not need it during inference anyway
        # and this class is for loading data for training - which is the only scenario where we need padding

        padding_size = self.sequence_length - len(encoding) + 1
        padding = [self.tokenizer.pad_token_id] * padding_size
        encoding = encoding + padding
        # The inputs to the transformer will be the offset sequence
        source_encoding = encoding[:-1]
        target_encoding = encoding[1:]

        source_token_ids = torch.tensor(source_encoding[: self.sequence_length], dtype=torch.int64)
        target_token_ids = torch.tensor(target_encoding[: self.sequence_length], dtype=torch.int64)
        target_mask = target_token_ids != self.tokenizer.pad_token_id
        if self.loss_masking == "finetuning":
            target_mask[: len(prompt_token_ids)] = False
        out = {
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "target_mask": target_mask,
            "prediction_task": "high_median_prediction",
            "source": record["source"],
        }
        return out
