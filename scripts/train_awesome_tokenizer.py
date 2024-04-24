from datasets import load_dataset, concatenate_datasets
from midi_tokenizers.one_time_tokenizer import NoLossTokenizer
from midi_trainable_tokenizers.awesome_midi_tokenzier import AwesomeMidiTokenizer

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
        base_tokenizer=base_tokenizer, max_vocab_size=max_vocab_size, max_token_length=max_token_length
    )
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")
    giant_dataset = load_dataset("roszcz/giant-midi-sustain-v2", split="train")
    dataset = concatenate_datasets([dataset, giant_dataset])

    tokenizer.train(dataset)
    tokenizer.save_tokenizer(path=path)


if __name__ == "__main__":
    min_time_unit = 0.01
    n_velocity_bins = 32
    max_token_length = 32
    max_vocab_size = 30000

    # Create the filename
    path = "pre-trained/awesome_tokenizers/awesome-tokenizer-pretrained.json"
    train(
        path=path,
        min_time_unit=min_time_unit,
        n_velocity_bins=n_velocity_bins,
        max_token_length=max_token_length,
        max_vocab_size=max_vocab_size,
    )
