import json

import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset
from midi_tokenizers_generation.tokenizer_generator import generate_tokenizer


def main():
    dataset_names = ["NoLossTokDataset", "OneTimeTokDataset"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    dataset_split = st.selectbox(label="split", options=["train", "test", "validation"])

    dataset = load_dataset(f"./{dataset_name}", name="debugging", split=dataset_split, trust_remote_code=True)

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

    streamlit_pianoroll.from_fortepyan(piece=piece)

    with st.expander(label="tokens"):
        st.write(record["note_tokens"])


if __name__ == "__main__":
    main()
