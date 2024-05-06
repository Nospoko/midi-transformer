from datasets import DatasetInfo
from midi_tokenizers.one_time_tokenizer import NoLossTokenizer

from tokenized_midi_datasets import TokenizedMidiDataset

_DESC = """
Dataset with midi files, tokenzied using NoLossTokenizer, with records of equal size.
"""


class ExponentialTimeTokenDataset(TokenizedMidiDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def load_tokenizer(self):
        return NoLossTokenizer(**self.config.tokenizer_parameters)
