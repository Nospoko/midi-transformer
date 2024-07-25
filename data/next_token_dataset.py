from typing import Literal

import torch
import numpy as np
from datasets import Dataset as HuggingFaceDataset

from data.dataset import MidiDataset
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


class NextTokenDataset(MidiDataset):
    """
    A PyTorch Dataset class for generating next token predictions from tokenized MIDI datasets.

    Attributes:
        dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
        tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        sequence_length: int,
        loss_masking: Literal["finetuning", "pretraining"] = "pretraining",
    ):
        """
        Initialize the NextTokenDataset.

        Parameters:
            dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
            tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
        """
        super().__init__(dataset=dataset, tokenizer=tokenizer, loss_masking=loss_masking)
        self.sequence_length = sequence_length
        self.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(4)))

    def __getitem__(self, idx: int) -> dict:
        """
        Randomly sample a record at the specified index and prepare it for next token prediction.

        Parameters:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: A dictionary containing the source and target token ids for next token prediction.
        """
        record = self.dataset[idx]
        # Random samplig ftw!!!!!!!!!
        full_encoding = record["note_token_ids"]
        n_tokens = len(full_encoding)

        if n_tokens < self.sequence_length:
            padding = [self.tokenizer.pad_token_id] * (self.sequence_length - n_tokens)
            full_encoding = full_encoding + padding
            n_tokens = self.sequence_length

        start = self.rs.randint(n_tokens - self.sequence_length + 1)
        encoding = full_encoding[start : start + self.sequence_length + 1]

        # The inputs to the transformer will be the offset sequence
        source_encoding = encoding[:-1]
        target_encoding = encoding[1:]

        source_token_ids = torch.tensor(source_encoding, dtype=torch.int64)
        target_token_ids = torch.tensor(target_encoding, dtype=torch.int64)
        target_mask = target_token_ids != self.tokenizer.pad_token_id

        out = {
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "target_mask": target_mask,
            "source": record["source"],
        }
        return out
