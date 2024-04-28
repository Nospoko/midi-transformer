import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset

from data.augmentation import pitch_shift, change_speed


def main():
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")

    record_id = st.number_input(label="record_id", value=0)
    record = dataset[record_id]

    piece = ff.MidiPiece.from_huggingface(record=record)
    df = piece.df.copy()
    st.write("we use `shift_threshold` = 5")
    shift = st.number_input(label="shift_threshold", value=5)
    st.write("we use 0.8 < `factor` < 1.2")
    factor = st.number_input(label="speed change factor", value=1.0)

    augmented_notes, shift = pitch_shift(df=df, shift_threshold=shift)
    augmented_notes, factor = change_speed(df=augmented_notes, factor=factor)
    aug_piece = ff.MidiPiece(df=augmented_notes)
    piece_columns = st.columns(2)
    with piece_columns[0]:
        st.write("Original:")
        streamlit_pianoroll.from_fortepyan(piece=piece)
    with piece_columns[1]:
        st.write("Augmented:")
        streamlit_pianoroll.from_fortepyan(piece=aug_piece)
        st.write(f"shift: {shift}, factor: {factor}")


if __name__ == "__main__":
    main()
