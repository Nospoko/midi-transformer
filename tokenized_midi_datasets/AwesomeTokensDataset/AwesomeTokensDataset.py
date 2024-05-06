from datasets import DatasetInfo
from hydra.utils import to_absolute_path
from midi_trainable_tokenizers.awesome_midi_tokenzier import AwesomeMidiTokenizer

from tokenized_midi_datasets import TokenizedMidiDataset

_DESC = """
Dataset with midi files, tokenzied using AwesomeTokensTokenizer, with records of equal size.
"""


class AwesomeTokensDataset(TokenizedMidiDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def load_tokenizer(self):
        pretrained_path = to_absolute_path("pretrained/awesome_tokenizers/awesome-tokenizer-pretrained.json")
        tokenizer = AwesomeMidiTokenizer.from_file(pretrained_path)
        return tokenizer
