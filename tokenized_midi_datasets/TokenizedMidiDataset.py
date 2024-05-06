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
Dataset with midi files, tokenzied using MidiTokenizer with records of equal size.
"""


class TokenizedMidiDataset(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    BUILDER_CONFIG_CLASS = TokenizedMidiDatasetConfig
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "basic-mid"

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        base = datasets.load_dataset(self.config.base_dataset_name)

        other_datasets = [datasets.load_dataset(path, split="train") for path in self.config.extra_datasets]
        other_datasets.append(base["train"])

        dataset = datasets.concatenate_datasets(other_datasets)
        dataset = augment_dataset(
            dataset=dataset,
            augmentation_probability=self.config.augmentation_probability,
            augmentation_repetitions=self.config.augmentation_repetitions,
        )

        # This will enable multiprocessing in load_dataset()
        n_shards = 12
        train_shards = [dataset.shard(n_shards, it) for it in range(n_shards)]

        return [
            datasets.SplitGenerator(name=Split.TRAIN, gen_kwargs={"dataset_shards": train_shards}),
            datasets.SplitGenerator(name=Split.TEST, gen_kwargs={"dataset_shards": [base["test"]]}),
            datasets.SplitGenerator(name=Split.VALIDATION, gen_kwargs={"dataset_shards": [base["validation"]]}),
        ]

    def tokenize_piece(self, piece: ff.MidiPiece):
        notes = piece.df
        tokens = self.tokenizer.encode(notes=notes)
        new_record = {
            "note_token_ids": tokens,
            "source": json.dumps(piece.source),
        }
        return new_record

    def piece_to_records(self, piece: ff.MidiPiece) -> list[dict]:
        tokenized_record = self.tokenize_piece(piece)
        n_tokens = len(tokenized_record["note_token_ids"])
        # better practice than setting a global random state
        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(4)))

        n_samples = 1 + (n_tokens - self.config.sequence_length) // self.config.sequence_step
        # uniform distribution, piece should be covered almost entirely
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
        next_start = piece.df.start.shift(-1)
        silent_distance = next_start - piece.df.end

        # Seconds
        ids = silent_distance > self.config.pause_detection_threshold

        break_idxs = np.where(ids)[0]

        pieces = []

        start = 0
        for break_idx in break_idxs:
            finish = break_idx.item() + 1
            piece_part = piece[start:finish]

            if piece_part.size <= self.config.sequence_length:
                continue

            pieces.append(piece_part)
            start = finish

        return pieces

    def _generate_examples(self, dataset_shards: list[Dataset]):
        self.tokenizer = self.load_tokenizer()

        for shard_id, dataset in enumerate(dataset_shards):
            for it, record in enumerate(dataset):
                piece = ff.MidiPiece.from_huggingface(dict(record))

                pieces = self.filter_pauses(piece)
                chopped_sequences = sum([self.piece_to_records(piece) for piece in pieces], [])

                for jt, sequence in enumerate(chopped_sequences):
                    key = f"{it}_{jt}_{shard_id}"  # for some reason there was duplicate key error without shard_id
                    yield key, sequence

    @abstractmethod
    def load_tokenizer(self):
        pass
