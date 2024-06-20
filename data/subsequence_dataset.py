import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer

from data.dataset import MidiDataset

# For the future
placeholder_tokens = ["<SENTINEL_{idx}>" for idx in range(100)]
special_tokens = [
    "<CLS>",
    "<EOS>",
    "<PAD>",
    "<RANDOM>",
    "<PPP>",
    "<PP>",
    "<P>",
    "<MP>",
    "<MF>",
    "<F>",
    "<BASS>",
    "<TENOR>",
    "<ALTO>",
    "<SOPRANO>",
    "<TREBLE>",
    "<NO_RANDOM>",
    "<NO_PPP>",
    "<NO_PP>",
    "<NO_P>",
    "<NO_MP>",
    "<NO_MF>",
    "<NO_F>",
    "<NO_BASS>",
    "<NO_TENOR>",
    "<NO_ALTO>",
    "<NO_SOPRANO>",
    "<NO_TREBLE>",
] + placeholder_tokens

pitch_masks = ["bass", "tenor", "alto", "soprano", "treble"]
extraction_type_to_token_pair = {
    "bass": ("<BASS>", "<NO_BASS>"),
    "tenor": ("<TENOR>", "<NO_TENOR>"),
    "alto": ("<ALTO>", "<NO_ALTO>"),
    "soprano": ("<SOPRANO>", "<NO_SOPRANO>"),
    "treble": ("<TREBLE>", "<NO_TREBLE>"),
    "ppp": ("<PPP>", "<NO_PPP>"),
    "pp": ("<PP>", "<NO_PP>"),
    "p": ("<P>", "<NO_P>"),
    "mp": ("<MP>", "<NO_MP>"),
    "mf": ("<MF>", "<NO_MF>"),
    "f": ("<F>", "<NO_F>"),
}

extraction_type_to_range = {
    "bass": (21, 48),
    "tenor": (43, 81),
    "alto": (53, 84),
    "soprano": (60, 96),
    "treble": (60, 108),
    "ppp": (0, 30),
    "pp": (30, 50),
    "p": (50, 70),
    "mp": (70, 90),
    "mf": (90, 110),
    "f": (110, 127),
}


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
        src_encoding = self.tokenizer.encode(notes=pd.DataFrame(record["src_notes"]))
        tgt_encoding = self.tokenizer.encode(notes=pd.DataFrame(record["tgt_notes"]))
        extracted = record["extracted"]

        source_token_ids, target_token_ids = self.post_process(
            source_token_ids=src_encoding,
            target_token_ids=tgt_encoding,
            extracted=extracted,
        )

        out = {
            "source_token_ids": torch.tensor(source_token_ids, dtype=torch.int64),
            "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int64),
            "extracted": extracted,
            "source": record["source"],
        }
        return out
