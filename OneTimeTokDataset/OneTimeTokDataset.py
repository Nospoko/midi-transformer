import json
from typing import List
from dataclasses import field, dataclass

import datasets
import numpy as np
import fortepyan as ff
from tqdm import tqdm
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from datasets import Split, Dataset, DatasetInfo, BuilderConfig, GeneratorBasedBuilder

_DESC = """
Dataset with tokenized midi files, split to equal size.
"""


@dataclass
class TokenizedDatasetInfo(DatasetInfo):
    tokenizer_name: str = "OneTimeTokenizer"
    tokenizer_parameters: dict = field(default_factory=dict)


class OneTimeTokDatasetConfig(BuilderConfig):
    def __init__(
        self,
        base_dataset_name: str = "roszcz/maestro-v1-sustain",
        extra_datasets: list[str] = [],
        sequence_length: int = 64,
        sequence_step: int = 42,
        tokenizer_parameters: dict = {"eps": 0.001, "n_velocity_bins": 32},
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
        self.tokenizer_name = "OneTimeTokenzier"


class OneTimeTokDataset(GeneratorBasedBuilder):
    def _info(self) -> TokenizedDatasetInfo:
        tokenizer_info = {
            "tokenizer_name": self.config.tokenizer_name,
            "tokenizer_parameters": self.config.tokenizer_parameters,
        }
        # I found no better way of storing metadata :(
        return TokenizedDatasetInfo(description=json.dumps(tokenizer_info))

    BUILDER_CONFIG_CLASS = OneTimeTokDatasetConfig
    BUILDER_CONFIGS = [
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=64,
            sequence_step=16,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="giant-short",
        ),
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=64,
            sequence_step=16,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="basic-short",
        ),
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=128,
            sequence_step=16,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="giant-mid",
        ),
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=128,
            sequence_step=16,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="basic-mid",
        ),
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=["roszcz/giant-midi-sustain-v2"],
            sequence_length=256,
            sequence_step=32,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="giant-long",
        ),
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=256,
            sequence_step=32,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="basic-long",
        ),
        OneTimeTokDatasetConfig(
            base_dataset_name="roszcz/maestro-sustain-v2",
            extra_datasets=[],
            sequence_length=512,
            sequence_step=512,
            tokenizer_parameters={"eps": 0.001, "n_velocity_bins": 32},
            name="debugging",
        ),
    ]
    DEFAULT_CONFIG_NAME = "basic-mid"

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
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

    def fix_token_sequences(self, tokens: list[str]):
        """
        If there are NOTE_OFF tokens before NOTE_ON in the list,
        add NOTE_ON events at the beginning of the token list.
        If there are NOTE_ON tokens that are left without NOTE_OFF, add them at the end as well.
        """
        pressed = np.zeros(shape=(110))
        # There are velocity tokens before each event - we know with which velocity the key
        # we are releasing was played
        current_velocity_token = "VELOCITY_0"
        for token in tokens:
            if "VELOCITY" in token:
                current_velocity_token = token

            if "NOTE_ON" in token:
                velocity = self.tokenizer.token_to_velocity_bin[current_velocity_token]
                pressed[self.tokenizer.token_to_pitch[token]] = velocity

            if "NOTE_OFF" in token:
                pitch = self.tokenizer.token_to_pitch[token]
                if pressed[pitch] == 0:
                    tokens = [current_velocity_token, self.tokenizer.pitch_to_on_token[pitch]] + tokens
                pressed[pitch] = 0
            for key, state in enumerate(pressed):
                if state > 0:
                    note_off_token = self.tokenizer.pitch_to_off_token[key]
                    velocity_token = self.tokenizer.velocity_bin_to_token[state]
                    tokens = tokens + [velocity_token, note_off_token]

        return tokens

    def tokenize_piece(self, piece: ff.MidiPiece):
        notes = piece.df
        tokens = self.tokenizer.tokenize(notes=notes)
        new_record = {
            "note_tokens": tokens,
            "source": piece.source,
        }
        return new_record

    def piece_to_records(self, piece: ff.MidiPiece) -> list[dict]:
        tokenized_record = self.tokenize_piece(piece)
        n_tokens = len(tokenized_record["note_tokens"])
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
            part = tokenized_record["note_tokens"][start:finish]

            # I still wonder if we should fix the sequences - it is very time-consuming.
            # Maybe take care of it during untokenization instead?
            #
            # part = self.fix_token_sequences(tokens=part)

            record = {
                "note_tokens": part,
                "source": tokenized_record["source"],
            }
            chopped_sequences.append(record)

        return chopped_sequences

    def filter_pauses(self, piece: ff.MidiPiece) -> list[ff.MidiPiece]:
        next_start = piece.df.start.shift(-1)
        silent_distance = next_start - piece.df.end

        # Seconds
        distance_threshold = 4

        ids = silent_distance > distance_threshold

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
        self.tokenizer = OneTimeTokenizer(**self.config.tokenizer_parameters)
        for dataset in dataset_shards:
            for it, record in tqdm(enumerate(dataset), total=len(dataset)):
                piece = ff.MidiPiece.from_huggingface(dict(record))

                pieces = self.filter_pauses(piece)
                chopped_sequences = sum([self.piece_to_records(piece) for piece in pieces], [])

                for jt, sequence in enumerate(chopped_sequences):
                    key = f"{it}_{jt}"
                    yield key, sequence
