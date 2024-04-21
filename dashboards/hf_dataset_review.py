import json

import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset
from midi_tokenizers_generation.tokenizer_generator import generate_tokenizer

from NoLossTokDataset.NoLossTokDataset import NoLossTokDataset
from OneTimeTokDataset.OneTimeTokDataset import OneTimeTokDataset


def main():
    dataset_names = ["NoLossTokDataset", "OneTimeTokDataset"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    configs = ["debugging", "giant-short", "basic-short", "giant-mid", "basic-mid", "giant-long", "basic-long"]

    config_name = st.selectbox(label="config name", options=configs)
    # Another way of accessing configs ...
    if dataset_name == "OneTimeTokDataset":
        config = OneTimeTokDataset.builder_configs[config_name]
    elif dataset_name == "NoLossTokDataset":
        config = NoLossTokDataset.builder_configs[config_name]
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])

    dataset = load_dataset(
        f"./{dataset_name}",
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

    tokenier_info = json.loads(dataset.description)
    tokenizer = generate_tokenizer(
        name=tokenier_info["tokenizer_name"],
        parameters=tokenier_info["tokenizer_parameters"],
    )
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
