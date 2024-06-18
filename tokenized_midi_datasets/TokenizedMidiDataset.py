import json
from abc import abstractmethod

import datasets
import numpy as np
import fortepyan as ff
from datasets import Split, Dataset, DatasetInfo, GeneratorBasedBuilder

from data.augmentation import augment_dataset
from tokenized_midi_datasets.TokenizedMidiDatasetConfig import BUILDER_CONFIGS, TokenizedMidiDatasetConfig

# NOTE: If you make some changes here, you might want to delete your huggingface cache
# at ~/.cache/huggingface/ to rebuild the datasets

_DESC = """
Dataset with MIDI files, tokenized using MidiTokenizer with records of equal size.
"""


class TokenizedMidiDataset(GeneratorBasedBuilder):
    """
    Dataset builder for tokenized MIDI datasets.

    This class is responsible for downloading, processing, and splitting MIDI datasets into train, test,
    and validation sets, applying augmentations, tokenizing MIDI files, and generating examples.
    """

    def _info(self) -> DatasetInfo:
        """
        Returns dataset metadata.

        Returns:
            DatasetInfo: Metadata about the dataset.
        """
        return DatasetInfo(description=_DESC)

    # Define the configuration class and available configurations
    BUILDER_CONFIG_CLASS = TokenizedMidiDatasetConfig
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "basic"

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        # Load the base dataset and additional datasets
        base = datasets.load_dataset(self.config.base_dataset_name)
        other_datasets = [datasets.load_dataset(path, split="train") for path in self.config.extra_datasets]
        other_datasets.append(base["train"])

        # Concatenate all datasets and apply augmentation
        dataset = datasets.concatenate_datasets(other_datasets)
        dataset = augment_dataset(
            dataset=dataset,
            max_pitch_shift=self.config.augmentation["max_pitch_shift"],
            speed_change_factors=self.config.augmentation["speed_change_factors"],
        )

        # Enable multiprocessing by splitting the dataset into shards
        n_shards = 12
        train_shards = [dataset.shard(n_shards, it) for it in range(n_shards)]

        return [
            datasets.SplitGenerator(name=Split.TRAIN, gen_kwargs={"dataset_shards": train_shards}),
            datasets.SplitGenerator(name=Split.TEST, gen_kwargs={"dataset_shards": [base["test"]]}),
            datasets.SplitGenerator(name=Split.VALIDATION, gen_kwargs={"dataset_shards": [base["validation"]]}),
        ]

    def tokenize_piece(self, piece: ff.MidiPiece) -> dict:
        """
        Tokenizes a MIDI piece.

        Parameters:
            piece (ff.MidiPiece): MIDI piece to tokenize.

        Returns:
            dict: Tokenized representation of the MIDI piece.
        """
        notes = piece.df
        tokens = self.tokenizer.encode(notes=notes)
        new_record = {
            "note_token_ids": tokens,
            "source": json.dumps(piece.source),
        }
        return new_record

    def piece_to_records(self, piece: ff.MidiPiece) -> list[dict]:
        """
        Splits a tokenized MIDI piece into smaller records of fixed length.

        Parameters:
            piece (ff.MidiPiece): Tokenized MIDI piece to split.

        Returns:
            list[dict]: List of records containing fixed-length sequences of tokens.
        """
        tokenized_record = self.tokenize_piece(piece)
        n_tokens = len(tokenized_record["note_token_ids"])
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(4)))

        n_samples = 1 + (n_tokens - self.config.sequence_length) // self.config.sequence_step
        piece_idxs = range(n_tokens - self.config.sequence_length)
        start_points = rs.choice(piece_idxs, size=n_samples, replace=False)

        chopped_sequences = []
        for start in start_points:
            start = int(start)
            finish = start + self.config.sequence_length
            part = tokenized_record["note_token_ids"][start:finish]
            record = {
                "note_token_ids": part,
                "source": tokenized_record["source"],
            }
            chopped_sequences.append(record)

        return chopped_sequences

    def filter_pauses(self, piece: ff.MidiPiece) -> list[ff.MidiPiece]:
        """
        Splits a MIDI piece into smaller pieces based on pauses (silent periods).

        Parameters:
            piece (ff.MidiPiece): MIDI piece to split based on pauses.

        Returns:
            list[ff.MidiPiece]: List of MIDI pieces without long pauses.
        """
        next_start = piece.df.start.shift(-1)
        silent_distance = next_start - piece.df.end
        ids = silent_distance > self.config.pause_detection_threshold
        break_idxs = np.where(ids)[0]

        pieces = []
        start = 0
        for break_idx in break_idxs:
            finish = break_idx.item() + 1
            piece_part = piece[start:finish]
            if piece_part.size <= self.config.sequence_length // 4:
                continue
            pieces.append(piece_part)
            start = finish

        return pieces

    def _generate_examples(self, dataset_shards: list[Dataset]):
        """
        Generates examples from the dataset shards.

        Parameters:
            dataset_shards (list[Dataset]): List of dataset shards to generate examples from.

        Yields:
            dict: Key-value pairs representing each example.
        """
        self.tokenizer = self.load_tokenizer()

        for shard_id, dataset in enumerate(dataset_shards):
            for it, record in enumerate(dataset):
                piece = ff.MidiPiece.from_huggingface(dict(record))
                pieces = [piece]  # self.filter_pauses(piece)
                chopped_sequences = sum([self.piece_to_records(piece) for piece in pieces], [])
                for jt, sequence in enumerate(chopped_sequences):
                    key = f"{it}_{jt}_{shard_id}"
                    yield key, sequence

    @abstractmethod
    def load_tokenizer(self):
        """
        Abstract method to load the tokenizer. This method must be implemented by subclasses.
        """
        pass
