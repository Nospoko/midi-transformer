from data.dataset import MidiDataset
from datasets import Dataset as HuggingFaceDataset
from omegaconf import DictConfig
from midi_tokenizers.midi_tokenizer import MidiTokenizer

import torch 


class NextTokenDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        dataset_cfg: DictConfig,
    ):
        super().__init__(
            dataset=dataset,
            dataset_cfg=dataset_cfg,
        )
        
    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        tokens = record["notes"]
        encoding = [self.tokenizer.token_to_id[token] for token in tokens]

        # the inputs to the transformer will be the offset sequence
        source_token_ids = encoding[:-1]
        target_token_ids = encoding[1:]


        out = {
            "source_token_ids": torch.tensor(source_token_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int64),
        }
        return out

        
        