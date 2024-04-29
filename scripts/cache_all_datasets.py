from datasets import load_dataset


def main():
    dataset_names = ["AwesomeTokensDataset", "ExponentialTimeTokenDataset"]
    config_names = [
        "giant-mid-coarse",
        "giant-long-coarse",
        "giant-mid-coarse-augmented",
        "giant-long-coarse-augmented",
        "colossal-mid-coarse-augmented",
        "colossal-long-coarse-augmented",
    ]

    for dataset_name in dataset_names:
        for config_name in config_names:
            load_dataset(
                path=f"tokenized_midi_datasets/{dataset_name}",
                name=config_name,
                trust_remote_code=True,
            )


if __name__ == "__main__":
    main()
