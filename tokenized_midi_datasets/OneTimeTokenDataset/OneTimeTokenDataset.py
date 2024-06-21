from datasets import DatasetInfo
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer

from tokenized_midi_datasets import TokenizedMidiDataset

_DESC = """
Dataset with midi files, tokenzied using OneTimeTokenizer, with records of equal size.
"""


class OneTimeTokenDataset(TokenizedMidiDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def load_tokenizer(self):
        tokenizer = OneTimeTokenizer(**self.config.tokenizer_parameters)
        return tokenizer
