import fortepyan as ff
from tqdm import tqdm
from datasets import Dataset, DatasetInfo
from midi_tokenizers.one_time_tokenizer import NoLossTokenizer

from tokenized_midi_datasets import TokenizedMidiDataset

_DESC = """
Dataset with midi files, tokenzied using NoLossTokenizer, with records of equal size.
"""


class ExponentialTimeTokenDataset(TokenizedMidiDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def _generate_examples(self, dataset_shards: list[Dataset]):
        self.tokenizer = NoLossTokenizer(**self.config.tokenizer_parameters)
        for dataset in dataset_shards:
            for it, record in tqdm(enumerate(dataset), total=len(dataset)):
                piece = ff.MidiPiece.from_huggingface(dict(record))

                pieces = self.filter_pauses(piece)
                chopped_sequences = sum([self.piece_to_records(piece) for piece in pieces], [])

                for jt, sequence in enumerate(chopped_sequences):
                    key = f"{it}_{jt}"
                    yield key, sequence
