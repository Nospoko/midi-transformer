import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset
from midi_tokenizers import NoLossTokenizer

from data.next_token_dataset import NextTokenDataset
from tokenized_midi_datasets import ExponentialTimeTokenDataset


def main():
    st.write("MidiDataset review on an example of ExponentialTimeTokenDataset")
    tokenized_dataset_builder = ExponentialTimeTokenDataset
    tokenized_dataset = load_dataset(
        "tokenized_midi_datasets/ExponentialTimeTokenDataset",
        name="basic-no-overlap",
        split="test",
    )
    tokenizer_parameters = tokenized_dataset_builder.builder_configs["basic-no-overlap"].tokenizer_parameters
    tokenizer = NoLossTokenizer(**tokenizer_parameters)
    dataset_names = ["NextTokenDataset"]

    dataset_name = st.selectbox("midi dataset name", options=dataset_names)

    match dataset_name:
        case "NextTokenDataset":
            dataset = NextTokenDataset(
                dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )

    idx = st.number_input(label="record id", min_value=0, max_value=len(dataset))
    record = dataset[idx]

    source_ids = record["source_token_ids"]
    target_ids = record["target_token_ids"]

    source_notes = tokenizer.decode(source_ids)
    target_notes = tokenizer.decode(target_ids)

    source_piece = ff.MidiPiece(df=source_notes)
    target_piece = ff.MidiPiece(df=target_notes)

    pieces_columns = st.columns(2)

    with pieces_columns[0]:
        streamlit_pianoroll.from_fortepyan(source_piece)
    with pieces_columns[1]:
        streamlit_pianoroll.from_fortepyan(target_piece)
