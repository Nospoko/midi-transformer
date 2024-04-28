import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset
from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer

from tokenized_midi_datasets import OneTimeTokenDataset, ExponentialTimeTokenDataset


def main():
    dataset_names = ["ExponentialTimeTokenDataset", "OneTimeTokenDataset", "AwesomeTokensDataset"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    configs = [
        # Non-overlapping coarse
        "basic-no-overlap",
        "giant-no-overlap",
        "basic-no-overlap-augmented",
        "giant-no-overlap-augmented",
        # High-res datasets
        "giant-short",
        "basic-short",
        "giant-mid",
        "basic-mid",
        "giant-long",
        "basic-long",
        # Coarse Datasets
        "giant-short-coarse",
        "basic-short-coarse",
        "giant-mid-coarse",
        "basic-mid-coarse",
        "giant-long-coarse",
        "basic-long-coarse",
    ]

    config_name = st.selectbox(label="config name", options=configs)

    # Another way of accessing configs without storing metadata
    if dataset_name == "OneTimeTokenDataset":
        config = OneTimeTokenDataset.builder_configs[config_name]
        tokenizer = OneTimeTokenizer(**config.tokenizer_parameters)

    elif dataset_name == "ExponentialTimeTokenDataset":
        config = ExponentialTimeTokenDataset.builder_configs[config_name]
        tokenizer = NoLossTokenizer(**config.tokenizer_parameters)
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])

    dataset = load_dataset(
        f"tokenized_midi_datasets/{dataset_name}",
        name=config_name,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    total_tokens = config.sequence_length * dataset.num_rows
    st.write(f"rows: {dataset.num_rows}")
    st.write(f"total tokens: {total_tokens}")
    with st.expander("config"):
        st.write(config)

    idx = st.number_input(label="record_id", value=0, max_value=len(dataset))
    record = dataset[idx]

    with st.expander(label="source"):
        st.json(record["source"])

    notes = tokenizer.decode(record["note_token_ids"])
    piece = ff.MidiPiece(notes, source=record["source"])
    st.write(
        """
        If the tokenizer sees unmatched NOTE_OFF or NOTE_ON events
        it will treat them as if the notes were playing on the edges of the recording.
        """
    )
    streamlit_pianoroll.from_fortepyan(piece=piece)

    with st.expander(label="tokens"):
        st.write([tokenizer.vocab[token_id] for token_id in record["note_token_ids"]])


if __name__ == "__main__":
    main()
