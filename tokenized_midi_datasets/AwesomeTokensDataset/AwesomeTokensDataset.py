import fortepyan as ff
from tqdm import tqdm
from datasets import Dataset, DatasetInfo
from midi_trainable_tokenizers.awesome_midi_tokenzier import AwesomeMidiTokenizer

from tokenized_midi_datasets import TokenizedMidiDataset

_DESC = """
Dataset with midi files, tokenzied using OneTimeTokenizer, with records of equal size.
"""


class AwesomeTokensDataset(TokenizedMidiDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def _generate_examples(self, dataset_shards: list[Dataset]):
        pretrained_path = "pretrained/awesome_tokenizers/awesome-tokenizer-pretrained.json"
        self.tokenizer = AwesomeMidiTokenizer.from_file(pretrained_path)

        for shard_id, dataset in enumerate(dataset_shards):
            for it, record in tqdm(enumerate(dataset), total=len(dataset)):
                piece = ff.MidiPiece.from_huggingface(dict(record))

                pieces = self.filter_pauses(piece)
                chopped_sequences = sum([self.piece_to_records(piece) for piece in pieces], [])

                for jt, sequence in enumerate(chopped_sequences):
                    key = f"{it}_{jt}_{shard_id}"  # for some reason there was duplicate key error without shard_id
                    yield key, sequence
