import os
import json

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from dashboards.common.components import download_button


def main():
    models = ["midi-gpt2-302M-pretraining-2024-04-30-05-26", "midi-gpt2-333M-pretraining-2024-04-30-14-59"]
    model = st.selectbox(options=models, label="model")
    directory = f"tmp/{model}"
    with open(f"{directory}/file_descriptors.json", "r+") as file:
        file_descriptors = json.load(file)

    file_descriptors = pd.DataFrame.from_dict(file_descriptors, orient="index")
    composers = file_descriptors.composer.unique()

    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=0,
    )

    ids = file_descriptors.composer == selected_composer
    piece_titles = file_descriptors[ids].title.unique()

    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
    )

    ids = (file_descriptors.composer == selected_composer) & (file_descriptors.title == selected_title)
    selected_files = file_descriptors[ids]
    idx = st.number_input(label="idx", value=0, max_value=len(selected_files))
    path = f"{directory}/{selected_files.index[idx]}"

    piece = ff.MidiPiece.from_file(path)
    piece.source = selected_files.iloc[idx].to_dict()

    st.json(piece.source)
    st.write("whole model output")
    generated_notes_with_offset = piece.df[piece.df.start > piece.source["original end"]].copy()
    second_part = ff.MidiPiece(generated_notes_with_offset)

    # Model could have also add "NOTE_OFF" events to original sequence
    expanded_input_notes = piece.df[: -second_part.size].copy()
    expanded_piece = ff.MidiPiece(expanded_input_notes)
    streamlit_pianoroll.from_fortepyan(piece=expanded_piece, secondary_piece=second_part)

    try:
        with open(path, "rb") as file:
            download_button_str = download_button(
                object_to_download=file.read(),
                download_filename=path.split("/")[-1],
                button_text="Download midi with context",
            )
            st.markdown(download_button_str, unsafe_allow_html=True)
    except ValueError:
        print("Error with reading the file...")

    midi_path = f"tmp/{model}_{selected_files.index[idx]}.mid"
    generated_file = second_part.to_midi()

    try:
        generated_file.write(midi_path)
        with open(midi_path, "rb") as file:
            download_button_str = download_button(
                object_to_download=file.read(),
                download_filename=midi_path.split("/")[-1],
                button_text="Download generated midi",
            )
            st.markdown(download_button_str, unsafe_allow_html=True)
    finally:
        # make sure to always clean up
        os.unlink(midi_path)
