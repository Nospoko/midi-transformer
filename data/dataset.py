from abc import abstractmethod

from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer


class MidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: MidiTokenizer,
    ):
        super().__init__()

        # MidiTokenizer which was used during creation of the dataset
        self.tokenizer = tokenizer

        # Dataset with tokenized MIDI data
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass
