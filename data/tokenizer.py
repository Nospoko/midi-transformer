from pandas import DataFrame
from midi_tokenizers import ExponentialTimeTokenizer
from midi_trainable_tokenizers import AwesomeMidiTokenizer


class ExponentialTokenizer(ExponentialTimeTokenizer):
    def encode(
        self,
        notes: DataFrame,
        pad_to_size: int = None,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        encoding = super().encode(notes)

        if pad_to_size is None:
            padding_size = 0
        else:
            padding_size = pad_to_size - len(encoding) - len(prefix_tokens)

        suffix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.token_to_id["<PAD>"]] * padding_size

        return suffix_ids + encoding + padding


class AwesomeTokenizer(AwesomeMidiTokenizer):
    def encode(
        self,
        notes: DataFrame,
        pad_to_size: int = None,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        encoding = super().encode(notes)
        if pad_to_size is None:
            padding_size = 0
        else:
            padding_size = pad_to_size - len(encoding) - len(prefix_tokens)
        suffix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.token_to_id["<PAD>"]] * padding_size

        return suffix_ids + encoding + padding
