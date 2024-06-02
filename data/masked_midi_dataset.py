from typing import Literal

import torch
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer

from data.dataset import MidiDataset

# For the future
placeholder_tokens = ["<SENTINEL_{idx}>" for idx in range(100)]
special_tokens = [
    "<CLS>",
    "<EOS>",
    "<PAD>",
    "<MASK_RANDOM>",
    "<MASK_PPP>",
    "<MASK_PP>",
    "<MASK_P>",
    "<MASK_MP>",
    "<MASK_MF>",
    "<MASK_F>",
    "<MASK_BASS>",
    "<MASK_TENOR>",
    "<MASK_ALTO>",
    "<MASK_SOPRANO>",
    "<MASK_TREBLE>",
] + placeholder_tokens

MaskLiteral = Literal[
    "bass",
    "tenor",
    "alto",
    "soprano",
    "treble",
    "ppp",
    "pp",
    "p",
    "mp",
    "mf",
    "f",
]
pitch_masks = ["bass", "tenor", "alto", "soprano", "treble"]
mask_type_to_range = {
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


class MaskedMidiDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: MidiTokenizer,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
        )

    def __getitem__(self, idx: int) -> dict:
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

    def mask_sequence(
        self,
        tokens: list[str],
        mask_type: MaskLiteral,
    ):
        low, high = mask_type_to_range[mask_type]
        key = "pitch" if mask_type in pitch_masks else "velocity"

        notes = self.tokenizer.untokenize(tokens=tokens)
        extract_ids = (notes[key] >= low) & (notes[key] < high)

        notes_extracted = notes.iloc[extract_ids]
        notes_deprived = notes.iloc(~extract_ids)

        source = self.tokenizer.encode(notes=notes_deprived)
        target = self.tokenizer.encode(notes=notes_extracted)

        return source, target
