import torch
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset
from abc import abstractmethod

from midi_tokenizers.midi_tokenizer import MidiTokenizer
from object_generators.tokenizer_generator import TokenizerGenerator


class MidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        dataset_cfg: DictConfig,
    ):
        super().__init__()

        # Dataset with tokenized MIDI data
        self.dataset = dataset
        tokenzier_generator = TokenizerGenerator()
        
        # Dataset metadata
        self.dataset_cfg = dataset_cfg
        tokenizer_parameters = OmegaConf.to_container(dataset_cfg["tokenizer_params"])
        self.tokenizer = tokenzier_generator.generate_tokenizer(
            dataset_cfg["tokenizer_name"], 
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
