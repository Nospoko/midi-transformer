from datasets import load_dataset
from midi_tokenizers.one_time_tokenizer import NoLossTokenizer
from midi_trainable_tokenizers.awesome_midi_tokenzier import AwesomeMidiTokenizer

from data.masked_midi_dataset import special_tokens

# This is a script for training an AwesomeMidiTokenizer


def train(
    path: str,
    min_time_unit: float = 0.01,
    n_velocity_bins: int = 32,
    max_token_length: int = 32,
    max_vocab_size: int = 3000,
):
    base_tokenizer = NoLossTokenizer(min_time_unit=min_time_unit, n_velocity_bins=n_velocity_bins)
    tokenizer = AwesomeMidiTokenizer(
        base_tokenizer=base_tokenizer,
        max_vocab_size=max_vocab_size,
        max_token_length=max_token_length,
        special_tokens=special_tokens,
    )
    # Training the tokenizer on maestro will be enough for a good heristic
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")

    tokenizer.train(dataset)
    tokenizer.save_tokenizer(path=path)


if __name__ == "__main__":
    min_time_unit = 0.01
    n_velocity_bins = 32
    max_token_length = 32
    max_vocab_size = 30000

    # Create the filename
    path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
    train(
        path=path,
        min_time_unit=min_time_unit,
        n_velocity_bins=n_velocity_bins,
        max_token_length=max_token_length,
        max_vocab_size=max_vocab_size,
    )
