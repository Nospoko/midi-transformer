import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset

from data.dataset import MidiDataset
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer
from artifacts import get_source_extraction_token, get_target_extraction_token


class SubSequenceMidiDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: AwesomeTokenizer | ExponentialTokenizer,
        sequence_length: int,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
        )
        self.sequence_length = sequence_length

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        extraction_type = record["extraction_type"]
        source_prefix = get_source_extraction_token(extraction_type)
        target_prefix = get_target_extraction_token(extraction_type)
        source_token_ids = self.tokenizer.encode(
            notes=pd.DataFrame(record["source_notes"]),
            prefix_tokens=[source_prefix],
            pad_to_size=self.sequence_length,
        )
        target_token_ids = self.tokenizer.encode(
            notes=pd.DataFrame(record["target_notes"]),
            prefix_tokens=[target_prefix],
            pad_to_size=self.sequence_length,
        )

        out = {
            "source_token_ids": torch.tensor(source_token_ids[: self.sequence_length], dtype=torch.int64),
            "target_token_ids": torch.tensor(target_token_ids[: self.sequence_length], dtype=torch.int64),
            "extraction_type": extraction_type,
            "source": record["source"],
        }
        return out
