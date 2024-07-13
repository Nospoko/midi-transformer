from typing import Literal

import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset

from data.dataset import MidiDataset
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer
from artifacts import get_source_task_token, get_target_task_token


class SubSequenceMidiDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: AwesomeTokenizer | ExponentialTokenizer,
        sequence_length: int,
        loss_calculation_style: Literal["finetuning", "pretraining"] = "pretraining",
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
        )
        self.sequence_length = sequence_length
        self.loss_calculation_style = loss_calculation_style

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        extraction_type = record["extraction_type"]
        source_prefix = get_source_task_token(extraction_type)
        target_prefix = get_target_task_token(extraction_type)
        prompt_token_ids = self.tokenizer.encode(
            notes=pd.DataFrame(record["source_notes"]),
            prefix_tokens=[source_prefix],
        )
        target_token_ids = self.tokenizer.encode(
            notes=pd.DataFrame(record["target_notes"]),
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
        if self.loss_calculation_style == "finetuning":
            target_mask[: len(prompt_token_ids)] = False
        out = {
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "target_mask": target_mask,
            "extraction_type": extraction_type,
            "source": record["source"],
        }
        return out
