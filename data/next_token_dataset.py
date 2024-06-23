import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer

from data.dataset import MidiDataset


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
        tokenizer: MidiTokenizer,
        sequence_length: int,
    ):
        """
        Initialize the NextTokenDataset.

        Parameters:
            dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
            tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
        """
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
        )
        self.sequence_length = sequence_length

    def post_process(self, source_token_ids: list[int], target_token_ids: list[int]):
        """Append suffixes (padding) for source and target"""

        src_pad_size = self.sequence_length - len(source_token_ids)
        tgt_pad_size = self.sequence_length - len(target_token_ids)
        src_suffix = [self.tokenizer.token_to_id["<PAD>"]] * src_pad_size
        tgt_suffix = [self.tokenizer.token_to_id["<PAD>"]] * tgt_pad_size

        return source_token_ids + src_suffix, target_token_ids + tgt_suffix

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a record at the specified index and prepare it for next token prediction.

        Parameters:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: A dictionary containing the source and target token ids for next token prediction.
        """
        record = self.dataset[idx]
        notes = pd.DataFrame(record["notes"])
        encoding = self.tokenizer.encode(notes=notes)

        # The inputs to the transformer will be the offset sequence
        source_token_ids = encoding[:-1]
        target_token_ids = encoding[1:]
        source_token_ids, target_token_ids = self.post_process(
            source_token_ids=source_token_ids, target_token_ids=target_token_ids
        )

        out = {
            "source_token_ids": torch.tensor(source_token_ids[: self.sequence_length], dtype=torch.int64),
            "target_token_ids": torch.tensor(target_token_ids[: self.sequence_length], dtype=torch.int64),
            "source": record["source"],
        }
        return out
