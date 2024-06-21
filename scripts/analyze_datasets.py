import re

from datasets import Dataset, load_dataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer
from midi_trainable_tokenizers.awesome_midi_tokenzier import AwesomeMidiTokenizer

from data.masked_midi_dataset import special_tokens

awesome_tokenzier_path = "pretrained/awesome_tokenizers/awesome-tokenizer-0.01-32.json"
dataset_to_tokenizer_map: dict[str, MidiTokenizer] = {
    "ExponentialTimeTokenDataset": ExponentialTimeTokenizer(
        min_time_unit=0.01,
        n_velocity_bins=32,
        special_tokens=special_tokens,
    ),
    "OneTimeTokenDataset": OneTimeTokenizer(
        min_time_unit=0.01,
        n_velocity_bins=32,
        special_tokens=special_tokens,
    ),
    "AwesomeTokensDataset": AwesomeMidiTokenizer.from_file(path=awesome_tokenzier_path),
}


def load_no_overlap_datasets(dataset_name: str) -> tuple[Dataset, Dataset]:
    datasets_directory = "tokenized_midi_datasets"
    basic_dataset = load_dataset(
        f"{datasets_directory}/{dataset_name}",
        name="basic-no-overlap-augmented",
        trust_remote_code=True,
        num_proc=8,
    )
    giant_dataset = load_dataset(
        f"{datasets_directory}/{dataset_name}",
        name="giant-no-overlap",
        trust_remote_code=True,
        num_proc=8,
    )
    return basic_dataset, giant_dataset


def print_num_tokens(dataset_name: str):
    basic_dataset, giant_dataset = load_no_overlap_datasets(dataset_name=dataset_name)
    sequence_length = len(basic_dataset["train"][0]["note_token_ids"])
    num_tokens_basic = basic_dataset["train"].num_rows * sequence_length

    # Long and basic have the same validation and test splits
    num_tokens_val = basic_dataset["validation"].num_rows * sequence_length
    num_tokens__test = basic_dataset["test"].num_rows * sequence_length

    num_tokens_giant = giant_dataset["train"].num_rows * sequence_length

    print(f"Basic {dataset_name} augmented dataset tokens:")
    print(f"\ttrain: {num_tokens_basic}\n\ttest: {num_tokens__test}\n\tvalidation:{num_tokens_val}")
    print(f"Giant {dataset_name} dataset tokens:")
    print(f"\ttrain: {num_tokens_giant}\n\ttest: {num_tokens__test}\n\tvalidation:{num_tokens_val}")
    print(f"Colossal {dataset_name} dataset tokens:")
    print(f"\ttrain: {num_tokens_giant}\n\ttest: {num_tokens__test}\n\tvalidation:{num_tokens_val}")
    # Ratio the same as in base_tokenzizer inside
    if dataset_name == "AwesomeTokensDataset":
        return

    tokenizer = dataset_to_tokenizer_map[dataset_name]
    basic_time_tokens = 0
    for record in basic_dataset["train"]:
        for token_id in record["note_token_ids"]:
            if re.search(".T$", tokenizer.vocab[token_id]) is not None:
                basic_time_tokens += 1

    basic_ratio = basic_time_tokens / num_tokens_basic

    print(f"Basic {dataset_name} time tokens to the rest ratio: {basic_ratio}")
    print(f"{tokenizer.name} vocab size: {tokenizer.vocab_size}")


def main():
    dataset_names = ["AwesomeTokensDataset", "ExponentialTimeTokenDataset", "OneTimeTokenDataset"]
    for name in dataset_names:
        print_num_tokens(name)
        print("\n")


if __name__ == "__main__":
    main()
