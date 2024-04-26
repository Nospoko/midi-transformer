import json

import datasets
import numpy as np
import fortepyan as ff
from datasets import Split, DatasetInfo, BuilderConfig, GeneratorBasedBuilder

# NOTE: If you make some changes here, you might want to delete your huggingface cache
# at ~/.cache/huggingface/ to rebuild the datasets

_DESC = """
Dataset with midi files, tokenzied using MidiTokenizer with records of equal size.
"""


class TokenizedMidiDatasetConfig(BuilderConfig):
    def __init__(
        self,
        base_dataset_name: str = "roszcz/maestro-v1-sustain",
        extra_datasets: list[str] = [],
        sequence_length: int = 64,
        sequence_step: int = 42,
        pause_detection_threshold: int = 4,
        tokenizer_parameters: dict = {"min_time_unit": 0.001, "n_velocity_bins": 32},
        **kwargs,
    ):
        super().__init__()
        # Version history:
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)

        self.base_dataset_name: str = base_dataset_name
        self.extra_datasets: list[str] = extra_datasets
        self.sequence_length: int = sequence_length
        self.sequence_step: int = sequence_step
        self.tokenizer_parameters = tokenizer_parameters
        self.pause_detection_threshold = pause_detection_threshold


class TokenizedMidiDataset(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    BUILDER_CONFIG_CLASS = TokenizedMidiDatasetConfig
    BUILDER_CONFIGS = [
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=256,
            sequence_step=32,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="giant-short",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=256,
            sequence_step=32,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="basic-short",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=512,
            sequence_step=64,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="giant-mid",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=512,
            sequence_step=64,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="basic-mid",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=1024,
            sequence_step=64,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="giant-long",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=1024,
            sequence_step=64,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="basic-long",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=512,
            sequence_step=512,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="basic-no-overlap",
        ),
        TokenizedMidiDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=512,
            sequence_step=512,
            pause_detection_threshold=4,
            tokenizer_parameters={"min_time_unit": 0.001, "n_velocity_bins": 32},
            name="giant-no-overlap",
        ),
    ]
    DEFAULT_CONFIG_NAME = "basic-mid"

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        base = datasets.load_dataset(self.config.base_dataset_name)

        other_datasets = [datasets.load_dataset(path, split="train") for path in self.config.extra_datasets]
        other_datasets.append(base["train"])

        dataset = datasets.concatenate_datasets(other_datasets)

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
