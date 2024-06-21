import torch
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

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a record at the specified index and prepare it for next token prediction.

        Parameters:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: A dictionary containing the source and target token ids for next token prediction.
        """
        record = self.dataset[idx]
        encoding = record["note_token_ids"]

        # The inputs to the transformer will be the offset sequence
        source_token_ids = encoding[:-1]
        target_token_ids = encoding[1:]

        out = {
            "source_token_ids": torch.tensor(source_token_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int64),
        }
        return out
