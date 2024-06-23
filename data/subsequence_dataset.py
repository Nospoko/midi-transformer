import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer

from data.dataset import MidiDataset
from artifacts import extraction_type_to_token_pair


class SubSequenceMidiDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: MidiTokenizer,
        sequence_length: int,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
        )
        self.sequence_length = sequence_length

    def post_process(self, source_token_ids: list[int], target_token_ids: list[int], extracted: str):
        """Append prefixes and suffixes (padding) for source and target"""
        prefix_tokens = [extraction_type_to_token_pair[extraction_type] for extraction_type in extracted]

        src_prefix = [self.tokenizer.token_to_id[prefix_token[1]] for prefix_token in prefix_tokens]
        tgt_prefix = [self.tokenizer.token_to_id[prefix_token[0]] for prefix_token in prefix_tokens]

        src_pad_size = self.sequence_length - len(source_token_ids) - len(src_prefix)
        tgt_pad_size = self.sequence_length - len(target_token_ids) - len(tgt_prefix)
        src_suffix = [self.tokenizer.token_to_id["<PAD>"]] * src_pad_size
        tgt_suffix = [self.tokenizer.token_to_id["<PAD>"]] * tgt_pad_size

        return src_prefix + source_token_ids + src_suffix, tgt_prefix + target_token_ids + tgt_suffix

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        src_encoding = self.tokenizer.encode(notes=pd.DataFrame(record["source_notes"]))
        tgt_encoding = self.tokenizer.encode(notes=pd.DataFrame(record["target_notes"]))
        extracted = record["extracted"]

        source_token_ids, target_token_ids = self.post_process(
            source_token_ids=src_encoding,
            target_token_ids=tgt_encoding,
            extracted=extracted,
        )

        out = {
            "source_token_ids": torch.tensor(source_token_ids[: self.sequence_length], dtype=torch.int64),
            "target_token_ids": torch.tensor(target_token_ids[: self.sequence_length], dtype=torch.int64),
            "extracted": extracted,
            "source": record["source"],
        }
        return out
