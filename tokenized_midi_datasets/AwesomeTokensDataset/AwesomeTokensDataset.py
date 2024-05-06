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
        tokenizer_parameters = self.config.tokenizer_parameters
        n_velocity_bins = tokenizer_parameters["n_velocity_bins"]
        min_time_unit = tokenizer_parameters["min_time_unit"]
        tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
        pretrained_path = to_absolute_path(tokenizer_path)
        tokenizer = AwesomeMidiTokenizer.from_file(pretrained_path)
        return tokenizer
