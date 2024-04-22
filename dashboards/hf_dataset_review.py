import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset
from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer

from OneTimeTokenDataset.OneTimeTokenDataset import OneTimeTokenDataset
from ExponentialTimeTokenDataset.ExponentialTimeTokenDataset import ExponentialTimeTokenDataset


def main():
    dataset_names = ["ExponentialTimeTokenDataset", "OneTimeTokenDataset"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    configs = ["debugging", "giant-short", "basic-short", "giant-mid", "basic-mid", "giant-long", "basic-long"]

    config_name = st.selectbox(label="config name", options=configs)

    # Another way of accessing configs ...
    if dataset_name == "OneTimeTokenDataset":
        config = OneTimeTokenDataset.builder_configs[config_name]
        tokenizer = OneTimeTokenizer(**config.tokenizer_parameters)

    elif dataset_name == "ExponentialTimeTokenDataset":
        config = ExponentialTimeTokenDataset.builder_configs[config_name]
        tokenizer = NoLossTokenizer(**config.tokenizer_parameters)
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])

    dataset = load_dataset(
        f"{dataset_name}",
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

    notes = tokenizer.untokenize(record["note_tokens"])
    piece = ff.MidiPiece(notes, source=record["source"])
    st.write(
        """
        If the tokenizer sees unmatched NOTE_OFF or NOTE_ON events
        it will treat them as if the notes were playing on the edges of the recording.
        """
    )
    streamlit_pianoroll.from_fortepyan(piece=piece)

    with st.expander(label="tokens"):
        st.write(record["note_tokens"])


if __name__ == "__main__":
    main()
