import json
from abc import abstractmethod

from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset
from object_generators.tokenizer_generator import TokenizerGenerator


class MidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
    ):
        super().__init__()

        # Dataset with tokenized MIDI data
        self.dataset = dataset
        tokenizer_info = json.loads(dataset.description)
        tokenizer_name = tokenizer_info["tokenizer_name"]
        tokenizer_parameters = tokenizer_info["tokenizer_parameters"]

        tokenzier_generator = TokenizerGenerator()

        self.tokenizer = tokenzier_generator.generate_tokenizer(
            tokenizer_name,
            tokenizer_parameters,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass
