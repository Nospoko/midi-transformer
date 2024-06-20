from abc import abstractmethod

import datasets
import numpy as np
import fortepyan as ff
from datasets import Split, Dataset, DatasetInfo, GeneratorBasedBuilder

from data.augmentation import augment_dataset
from downstream_task_datasets.SubSequenceDatasetConfig import BUILDER_CONFIGS, SubSequenceDatasetConfig

# NOTE: If you make some changes here, you might want to delete your huggingface cache
# at ~/.cache/huggingface/ to rebuild the datasets

_DESC = """
Dataset with MIDI files, divided into src_notes and tgt_notes with equal sum of notes.
"""


class SubSequenceDataset(GeneratorBasedBuilder):
    """
    Dataset builder for sub-sequence-prediction MIDI datasets.

    This class is responsible for downloading, processing, and splitting MIDI datasets into train, test,
    and validation sets, applying augmentations, seperating input and target sequences and generating examples.
    """

    def _info(self) -> DatasetInfo:
        """
        Returns dataset metadata.

        Returns:
            DatasetInfo: Metadata about the dataset.
        """
        return DatasetInfo(description=_DESC)

    # Define the configuration class and available configurations
    BUILDER_CONFIG_CLASS = SubSequenceDatasetConfig
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "basic-no-overlap"

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

    def piece_to_records(self, piece: ff.MidiPiece) -> list[dict]:
        """
        Splits a tokenized MIDI piece into smaller records of fixed length.

        Parameters:
            piece (ff.MidiPiece): Tokenized MIDI piece to split.

        Returns:
            list[dict]: List of records containing fixed-length sequences of tokens.
        """
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(4)))
        n_notes = len(piece.df.pitch)
        # Some sequences might be too short
        if n_notes < self.config.notes_per_record:
            return []
        n_samples = 1 + (n_notes - self.config.notes_per_record) // self.config.sequence_step
        piece_idxs = range(n_notes - self.config.notes_per_record)
        start_points = rs.choice(piece_idxs, size=n_samples, replace=False)

        prepared_records = []
        for start in start_points:
            start = int(start)
            finish = start + self.config.notes_per_record
            part = piece[start:finish]
            record = self.create_record(part)
            prepared_records.append(record)

        return prepared_records

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
        if len(break_idxs) == 0:
            return [piece]

        pieces = []
        start = 0
        for break_idx in break_idxs:
            finish = break_idx.item() + 1
            piece_part = piece[start:finish]
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
        for shard_id, dataset in enumerate(dataset_shards):
            for it, record in enumerate(dataset):
                piece = ff.MidiPiece.from_huggingface(dict(record))
                pieces = self.filter_pauses(piece)
                all_records = sum([self.piece_to_records(piece) for piece in pieces], [])
                for jt, sequence in enumerate(all_records):
                    key = f"{it}_{jt}_{shard_id}"
                    yield key, sequence

    @abstractmethod
    def create_record(piece: ff.MidiPiece):
        """
        Abstract method to divide a piece into two sub-sequences. This method must be implemented by subclasses.
        """
        pass
